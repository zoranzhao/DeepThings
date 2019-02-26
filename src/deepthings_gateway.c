#include "deepthings_gateway.h"
#include "ftp.h"
#include "inference_engine_helper.h"
#include "frame_partitioner.h"
#include "reuse_data_serialization.h"
#if DEBUG_TIMING
static double start_time;
static double acc_time[MAX_EDGE_NUM];
static uint32_t acc_frames[MAX_EDGE_NUM];
#endif
#if DEBUG_COMMU_SIZE
static double commu_size;
#endif

device_ctxt* deepthings_gateway_init(uint32_t N, uint32_t M, uint32_t fused_layers, char* network, char* weights, uint32_t total_edge_number, const char** addr_list){
   device_ctxt* ctxt = init_gateway(total_edge_number, addr_list);
   cnn_model* model = load_cnn_model(network, weights);
   model->ftp_para = preform_ftp(N, M, fused_layers, model->net_para);
#if DATA_REUSE
   model->ftp_para_reuse = preform_ftp_reuse(model->net_para, model->ftp_para);
#endif
   ctxt->model = model;
   set_gateway_local_addr(ctxt, GATEWAY_LOCAL_ADDR);
   set_gateway_public_addr(ctxt, GATEWAY_PUBLIC_ADDR);
   set_total_frames(ctxt, FRAME_NUM);
   set_batch_size(ctxt, N*M);

   return ctxt;
}


#if DATA_REUSE
void notify_coverage(device_ctxt* ctxt, blob* task_input_blob, uint32_t cli_id){
   char data[20]="empty";
   blob* temp = new_blob_and_copy_data(cli_id, 20, (uint8_t*)data);
   copy_blob_meta(temp, task_input_blob);
   service_conn* conn;
   conn = connect_service(TCP, get_client_addr(cli_id, ctxt), WORK_STEAL_PORT);
   send_request("update_coverage", 20, conn);
   send_data(temp, conn);
   free_blob(temp);
   close_service_connection(conn);
}
#endif

/*Same implementation with result_gateway,just insert more profiling information*/
void* deepthings_result_gateway(void* srv_conn, void* arg){
   printf("result_gateway ... ... \n");
   device_ctxt* ctxt = (device_ctxt*)arg;
   service_conn *conn = (service_conn *)srv_conn;
   int32_t cli_id;
   int32_t frame_seq;
#if DEBUG_TIMING
   double total_time;
   uint32_t total_frames;
   double now;
   uint32_t i;
#endif
#if DEBUG_FLAG
   char ip_addr[ADDRSTRLEN];
   int32_t processing_cli_id;
   get_dest_ip_string(ip_addr, conn);
   processing_cli_id = get_client_id(ip_addr, ctxt);
   if(processing_cli_id < 0)
      printf("Client IP address unknown ... ...\n");
#endif
   blob* temp = recv_data(conn);
   cli_id = get_blob_cli_id(temp);
   frame_seq = get_blob_frame_seq(temp);
#if DEBUG_COMMU_SIZE
   commu_size = commu_size + temp->size;
#endif
#if DEBUG_FLAG
   printf("Result from %d: %s is for client %d, total number recved is %d\n", processing_cli_id, ip_addr, cli_id, ctxt->results_counter[cli_id]);
#endif
   enqueue(ctxt->results_pool[cli_id], temp);
   free_blob(temp);
   ctxt->results_counter[cli_id]++;
   if(ctxt->results_counter[cli_id] == ctxt->batch_size){
      temp = new_empty_blob(cli_id);
#if DEBUG_FLAG
      printf("Results for client %d are all collected in deepthings_result_gateway, update ready_pool\n", cli_id);
#endif
#if DEBUG_TIMING
      printf("Client %d, frame sequence number %d, all partitions are merged in deepthings_merge_result_thread\n", cli_id, frame_seq);
      now = sys_now_in_sec();
      /*Total latency*/
      acc_time[cli_id] = now - start_time;
      acc_frames[cli_id] = frame_seq + 1;
      total_time = 0;
      total_frames = 0;
      for(i = 0; i < ctxt->total_cli_num; i ++){
         if(acc_frames[i] > 0)
             printf("Avg latency for Client %d is: %f\n", i, acc_time[i]/acc_frames[i]);
         total_time = total_time + acc_time[i];
         total_frames = total_frames + acc_frames[i];
      }
      printf("Avg latency for all clients %f\n", total_time/total_frames);
#endif
#if DEBUG_COMMU_SIZE
      printf("Communication size at gateway is: %f\n", ((double)commu_size)/(1024.0*1024.0*FRAME_NUM));
#endif
      enqueue(ctxt->ready_pool, temp);
      free_blob(temp);
      ctxt->results_counter[cli_id] = 0;
   }

   return NULL;
}

void deepthings_collect_result_thread(void *arg){
   const char* request_types[]={"result_gateway"};
   void* (*handlers[])(void*, void*) = {deepthings_result_gateway};
   int result_service = service_init(RESULT_COLLECT_PORT, TCP);
   start_service(result_service, TCP, request_types, 1, handlers, arg);
   close_service(result_service);
}

void deepthings_merge_result_thread(void *arg){
   cnn_model* model = (cnn_model*)(((device_ctxt*)(arg))->model);
#ifdef NNPACK
   nnp_initialize();
   model->net->threadpool = pthreadpool_create(THREAD_NUM);
#endif
   blob* temp;
   int32_t cli_id;
   int32_t frame_seq;
   while(1){
      temp = dequeue_and_merge((device_ctxt*)arg);
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
static overlapped_tile_data* overlapped_data_pool[MAX_EDGE_NUM][PARTITIONS_MAX][FRAME_NUM];
/*
static bool partition_coverage[MAX_EDGE_NUM][PARTITIONS_MAX];
*/
void* recv_reuse_data_from_edge(void* srv_conn, void* arg){
   printf("collecting_reuse_data ... ... \n");
   service_conn *conn = (service_conn *)srv_conn;
   cnn_model* gateway_model = (cnn_model*)(((device_ctxt*)(arg))->model);

   int32_t cli_id;
   int32_t task_id;
   int32_t frame_seq;

   char ip_addr[ADDRSTRLEN];
   int32_t processing_cli_id;
   get_dest_ip_string(ip_addr, conn);
   processing_cli_id = get_client_id(ip_addr, (device_ctxt*)arg);
   if(processing_cli_id < 0)
      printf("Client IP address unknown ... ...\n");

   blob* temp = recv_data(conn);
   cli_id = get_blob_cli_id(temp);
   task_id = get_blob_task_id(temp);
   frame_seq = get_blob_frame_seq(temp);
#if DEBUG_COMMU_SIZE
   commu_size = commu_size + temp->size;
#endif

#if DEBUG_DEEP_GATEWAY
   printf("Overlapped data for client %d, task %d is collected from %d: %s, size is %d\n", cli_id, task_id, processing_cli_id, ip_addr, temp->size);
#endif
   if(overlapped_data_pool[cli_id][task_id][frame_seq] != NULL)
      free_self_overlapped_tile_data(gateway_model,  overlapped_data_pool[cli_id][task_id][frame_seq]);
   overlapped_data_pool[cli_id][task_id][frame_seq] = self_reuse_data_deserialization(gateway_model, task_id, (float*)temp->data, get_blob_frame_seq(temp));

   if(processing_cli_id != cli_id) notify_coverage((device_ctxt*)arg, temp, cli_id);
   free_blob(temp);

   return NULL;
}

void* send_reuse_data_to_edge(void* srv_conn, void* arg){
   printf("handing_out_reuse_data ... ... \n");
   service_conn *conn = (service_conn *)srv_conn;
   device_ctxt* ctxt = (device_ctxt*)arg;
   cnn_model* gateway_model = (cnn_model*)(ctxt->model);

   int32_t cli_id;
   int32_t task_id;
   uint32_t frame_num;
   blob* temp = recv_data(conn);
   cli_id = get_blob_cli_id(temp);
   task_id = get_blob_task_id(temp);
   frame_num = get_blob_frame_seq(temp);
   free_blob(temp);

#if DEBUG_DEEP_GATEWAY
   char ip_addr[ADDRSTRLEN];
   int32_t processing_cli_id;
   get_dest_ip_string(ip_addr, conn);
   processing_cli_id = get_client_id(ip_addr, ctxt);
   if(processing_cli_id < 0)
      printf("Client IP address unknown ... ...\n");
#endif

   blob* reuse_info_blob = recv_data(conn);
   bool* reuse_data_is_required = (bool*)(reuse_info_blob->data);

#if DEBUG_DEEP_GATEWAY
   printf("Overlapped data for client %d, task %d is required by %d: %s is \n", cli_id, task_id, processing_cli_id, ip_addr);
   print_reuse_data_is_required(reuse_data_is_required);
#endif
   uint32_t position;
   int32_t* adjacent_id = get_adjacent_task_id_list(gateway_model, task_id);

   for(position = 0; position < 4; position++){
      if(adjacent_id[position]==-1) continue;
      if(reuse_data_is_required[position]){
#if DEBUG_DEEP_GATEWAY
         printf("place_self_deserialized_data for client %d, task %d, the adjacent task is %d\n", cli_id, task_id, adjacent_id[position]);
#endif
         place_self_deserialized_data(gateway_model, adjacent_id[position], overlapped_data_pool[cli_id][adjacent_id[position]][frame_num]);
      }
   }
   free(adjacent_id);
   temp = adjacent_reuse_data_serialization(ctxt, task_id, frame_num, reuse_data_is_required);
   free_blob(reuse_info_blob);
   send_data(temp, conn);
#if DEBUG_COMMU_SIZE
   commu_size = commu_size + temp->size;
#endif
   free_blob(temp);

   return NULL;
}

#endif

void deepthings_work_stealing_thread(void *arg){
#if DATA_REUSE
   const char* request_types[]={"register_gateway", "cancel_gateway", "steal_gateway", "reuse_data", "request_reuse_data"};
   void* (*handlers[])(void*, void*) = {register_gateway, cancel_gateway, steal_gateway, recv_reuse_data_from_edge, send_reuse_data_to_edge};
#else
   const char* request_types[]={"register_gateway", "cancel_gateway", "steal_gateway"};
   void* (*handlers[])(void*, void*) = {register_gateway, cancel_gateway, steal_gateway};
#endif

   int wst_service = service_init(WORK_STEAL_PORT, TCP);
#if DATA_REUSE
   start_service(wst_service, TCP, request_types, 5, handlers, arg);
#else
   start_service(wst_service, TCP, request_types, 3, handlers, arg);
#endif
   close_service(wst_service);
}


void deepthings_gateway(uint32_t N, uint32_t M, uint32_t fused_layers, char* network, char* weights, uint32_t total_edge_number, const char** addr_list){
   device_ctxt* ctxt = deepthings_gateway_init(N, M, fused_layers, network, weights, total_edge_number, addr_list);
   sys_thread_t t3 = sys_thread_new("deepthings_work_stealing_thread", deepthings_work_stealing_thread, ctxt, 0, 0);
   sys_thread_t t1 = sys_thread_new("deepthings_collect_result_thread", deepthings_collect_result_thread, ctxt, 0, 0);
   sys_thread_t t2 = sys_thread_new("deepthings_merge_result_thread", deepthings_merge_result_thread, ctxt, 0, 0);
   exec_barrier(START_CTRL, TCP, ctxt);
#if DEBUG_TIMING
   start_time = sys_now_in_sec();
#endif
   sys_thread_join(t1);
   sys_thread_join(t2);
   sys_thread_join(t3);
}



