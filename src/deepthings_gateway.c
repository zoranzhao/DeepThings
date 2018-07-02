#include "deepthings_gateway.h"
#include "config.h"
#include "ftp.h"
#include "inference_engine_helper.h"
#include "frame_partitioner.h"
#include "reuse_data_serialization.h"

cnn_model* deepthings_gateway_init(){
   init_gateway();
   cnn_model* model = load_cnn_model((char*)"models/yolo.cfg", (char*)"models/yolo.weights");
   model->ftp_para = preform_ftp(PARTITIONS_H, PARTITIONS_W, FUSED_LAYERS, model->net_para);
#if DATA_REUSE
   model->ftp_para_reuse = preform_ftp_reuse(model->net_para, model->ftp_para);
#endif
   return model;
}

void* deepthings_result_gateway(void* srv_conn){
   printf("result_gateway ... ... \n");
   service_conn *conn = (service_conn *)srv_conn;
   int32_t cli_id;
#if DEBUG_FLAG
   char ip_addr[ADDRSTRLEN];
   int32_t processing_cli_id;
   inet_ntop(conn->serv_addr_ptr->sin_family, &(conn->serv_addr_ptr->sin_addr), ip_addr, ADDRSTRLEN);
   processing_cli_id = get_client_id(ip_addr);
   if(processing_cli_id < 0)
      printf("Client IP address unknown ... ...\n");
#endif
   blob* temp = recv_data(conn);
   cli_id = get_blob_cli_id(temp);
#if DEBUG_FLAG
   printf("Result from %d: %s is for client %d, total number recved is %d\n", processing_cli_id, ip_addr, cli_id, results_counter[cli_id]);
#endif
   enqueue(results_pool[cli_id], temp);
   free_blob(temp);
   results_counter[cli_id]++;
   if(results_counter[cli_id] == PARTITIONS_H*PARTITIONS_W){
      temp = new_empty_blob(cli_id);
#if DEBUG_FLAG
      printf("Results for client %d are all collected in deepthings_result_gateway, update ready_pool\n", cli_id);
#endif
      enqueue(ready_pool, temp);
      free_blob(temp);
      results_counter[cli_id] = 0;
   }

   return NULL;
}

void deepthings_collect_result_thread(void *arg){
   const char* request_types[]={"result_gateway"};
   void* (*handlers[])(void*) = {deepthings_result_gateway};
   int result_service = service_init(RESULT_COLLECT_PORT, TCP);
   start_service(result_service, TCP, request_types, 1, handlers);
   close_service(result_service);
}

void deepthings_merge_result_thread(void *arg){
   cnn_model* model = (cnn_model*)arg;
#ifdef NNPACK
   nnp_initialize();
   model->net->threadpool = pthreadpool_create(THREAD_NUM);
#endif
   blob* temp;
   int32_t cli_id;
   int32_t frame_seq;
   while(1){
      temp = dequeue_and_merge(model);
      cli_id = get_blob_cli_id(temp);
      frame_seq = get_blob_frame_seq(temp);
#if DEBUG_FLAG
      printf("Client %d, frame sequence number %d, all partitions are merged in deepthings_merge_result_thread\n", cli_id, frame_seq);
#endif
      float* fused_output = (float*)(temp->data);
      image_holder img = load_image_as_model_input(model, get_blob_frame_seq(temp));
      set_model_input(model, fused_output);
      forward_all(model, model->ftp_para->fused_layers);   
      draw_object_boxes(model, get_blob_frame_seq(temp));
      free_image_holder(model, img);
      free_blob(temp);
#if DEBUG_FLAG
      printf("Client %d, frame sequence number %d, finish processing\n", cli_id, frame_seq);
#endif
   }
#ifdef NNPACK
   pthreadpool_destroy(model->net->threadpool);
   nnp_deinitialize();
#endif
}


#if DATA_REUSE
/*static overlapped_tile_data overlapped_data_and_parameters[CLI_NUM][PARTITIONS_MAX][FUSED_LAYERS_MAX];*/
static blob* overlapped_data_pool[CLI_NUM][PARTITIONS_MAX];

void* recv_reuse_data_from_edge(void* srv_conn){
   printf("collecting_reuse_data ... ... \n");
   service_conn *conn = (service_conn *)srv_conn;
   int32_t cli_id;
   int32_t task_id;
#if DEBUG_DEEP_GATEWAY
   char ip_addr[ADDRSTRLEN];
   int32_t processing_cli_id;
   inet_ntop(conn->serv_addr_ptr->sin_family, &(conn->serv_addr_ptr->sin_addr), ip_addr, ADDRSTRLEN);
   processing_cli_id = get_client_id(ip_addr);
   if(processing_cli_id < 0)
      printf("Client IP address unknown ... ...\n");
#endif
   blob* temp = recv_data(conn);
   cli_id = get_blob_cli_id(temp);
   task_id = get_blob_task_id(temp);
#if DEBUG_DEEP_GATEWAY
   printf("Overlapped data for client %d, task %d is collected from %d: %s is \n", cli_id, task_id, processing_cli_id, ip_addr);
#endif
   overlapped_data_pool[cli_id][task_id] = temp;
   return NULL;
}

void* send_reuse_data_to_edge(void* srv_conn){
   printf("handing_out_reuse_data ... ... \n");
   service_conn *conn = (service_conn *)srv_conn;

   int32_t cli_id;
   int32_t task_id;
   blob* temp = recv_data(conn);
   cli_id = get_blob_cli_id(temp);
   task_id = get_blob_task_id(temp);
   free_blob(temp);

#if DEBUG_DEEP_GATEWAY
   char ip_addr[ADDRSTRLEN];
   int32_t processing_cli_id;
   inet_ntop(conn->serv_addr_ptr->sin_family, &(conn->serv_addr_ptr->sin_addr), ip_addr, ADDRSTRLEN);
   processing_cli_id = get_client_id(ip_addr);
   if(processing_cli_id < 0)
      printf("Client IP address unknown ... ...\n");
   printf("Overlapped data for client %d, task %d is required by %d: %s is \n", cli_id, task_id, processing_cli_id, ip_addr);
#endif

   temp = overlapped_data_pool[cli_id][task_id];
   send_data(temp, conn);
   free_blob(temp);

   return NULL;
}

#endif

void deepthings_work_stealing_thread(void *arg){
#if DATA_REUSE
   const char* request_types[]={"register_gateway", "cancel_gateway", "steal_gateway", "reuse_data", "request_reuse_data"};
   void* (*handlers[])(void*) = {register_gateway, cancel_gateway, steal_gateway, recv_reuse_data_from_edge, send_reuse_data_to_edge};
#else
   const char* request_types[]={"register_gateway", "cancel_gateway", "steal_gateway"};
   void* (*handlers[])(void*) = {register_gateway, cancel_gateway, steal_gateway};
#endif

   int wst_service = service_init(WORK_STEAL_PORT, TCP);
#if DATA_REUSE
   start_service(wst_service, TCP, request_types, 5, handlers);
#else
   start_service(wst_service, TCP, request_types, 3, handlers);
#endif
   close_service(wst_service);
}


void deepthings_gateway(){
   cnn_model* model = deepthings_gateway_init();

   sys_thread_t t3 = sys_thread_new("deepthings_work_stealing_thread", deepthings_work_stealing_thread, model, 0, 0);
   sys_thread_t t1 = sys_thread_new("deepthings_collect_result_thread", deepthings_collect_result_thread, model, 0, 0);
   sys_thread_t t2 = sys_thread_new("deepthings_merge_result_thread", deepthings_merge_result_thread, model, 0, 0);
   exec_barrier(START_CTRL, TCP);

   sys_thread_join(t1);
   sys_thread_join(t2);
   sys_thread_join(t3);
}



