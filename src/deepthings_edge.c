#include "deepthings_edge.h"
#include "ftp.h"
#include "inference_engine_helper.h"
#include "frame_partitioner.h"
#include "reuse_data_serialization.h"
#if DEBUG_COMMU_SIZE
static double commu_size;
#endif
#if DEBUG_TIMING
static double start_time;
#endif

device_ctxt* deepthings_edge_init(uint32_t N, uint32_t M, uint32_t fused_layers, char* network, char* weights, uint32_t edge_id){
   device_ctxt* ctxt = init_client(edge_id);
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
#if DEBUG_LOG
   char logname[20];
   sprintf(logname, "edge_%d.log", edge_id);
   FILE *log = fopen(logname, "w");
   fclose(log);
#endif
   return ctxt;
}

#if DATA_REUSE
void send_reuse_data(device_ctxt* ctxt, blob* task_input_blob){
   cnn_model* model = (cnn_model*)(ctxt->model);
   /*if task doesn't generate any reuse_data*/
   if(model->ftp_para_reuse->schedule[get_blob_task_id(task_input_blob)] == 1) return;

   service_conn* conn;

   blob* temp  = self_reuse_data_serialization(ctxt, get_blob_task_id(task_input_blob), get_blob_frame_seq(task_input_blob));
   conn = connect_service(TCP, ctxt->gateway_local_addr, WORK_STEAL_PORT);
   send_request("reuse_data", 20, conn);
#if DEBUG_DEEP_EDGE
   printf("send self reuse data for task %d:%d \n", get_blob_cli_id(task_input_blob), get_blob_task_id(task_input_blob)); 
#endif
/*#if DEBUG_LOG
   char logname[20];
   sprintf(logname, "edge_%d.log", ctxt->this_cli_id);
   FILE *log = fopen(logname, "a");
   fprintf(log, "Start sending self reuse data for task %d:%d at time %f\n", get_blob_cli_id(task_input_blob), get_blob_task_id(task_input_blob), sys_now_in_sec()-start_time); 
#endif*/
   copy_blob_meta(temp, task_input_blob);
   send_data(temp, conn);
/*#if DEBUG_LOG
   fprintf(log, "Finish sending self reuse data for task %d:%d at time %f\n", get_blob_cli_id(task_input_blob), get_blob_task_id(task_input_blob), sys_now_in_sec()-start_time); 
   fclose(log);
#endif*/
   free_blob(temp);
   close_service_connection(conn);
}

void request_reuse_data(device_ctxt* ctxt, blob* task_input_blob, bool* reuse_data_is_required){
   cnn_model* model = (cnn_model*)(ctxt->model);
   /*if task doesn't require any reuse_data*/
   if(model->ftp_para_reuse->schedule[get_blob_task_id(task_input_blob)] == 0) return;/*Task without any dependency*/
   if(!need_reuse_data_from_gateway(reuse_data_is_required)) return;/*Reuse data are all generated locally*/

   service_conn* conn;
   conn = connect_service(TCP, ctxt->gateway_local_addr, WORK_STEAL_PORT);
   send_request("request_reuse_data", 20, conn);
   char data[20]="empty";
   blob* temp = new_blob_and_copy_data(get_blob_task_id(task_input_blob), 20, (uint8_t*)data);
   copy_blob_meta(temp, task_input_blob);
   send_data(temp, conn);
   free_blob(temp);
#if DEBUG_DEEP_EDGE
   printf("Request reuse data for task %d:%d \n", get_blob_cli_id(task_input_blob), get_blob_task_id(task_input_blob)); 
#endif
/*#if DEBUG_LOG
   char logname[20];
   sprintf(logname, "edge_%d.log", ctxt->this_cli_id);
   FILE *log = fopen(logname, "a");
   fprintf(log, "Request reuse data for task %d, (Frame %d, Cli %d) at time %f\n", get_blob_task_id(task_input_blob),
            get_blob_frame_seq(task_input_blob), get_blob_cli_id(task_input_blob), sys_now_in_sec()-start_time); 
#endif*/
   temp = new_blob_and_copy_data(get_blob_task_id(task_input_blob), sizeof(bool)*4, (uint8_t*)reuse_data_is_required);
   copy_blob_meta(temp, task_input_blob);
   send_data(temp, conn);
   free_blob(temp);
   temp = recv_data(conn);
/*#if DEBUG_LOG
   fprintf(log, "Recvd reuse data for task %d, (Frame %d, Cli %d) at time %f\n", get_blob_task_id(task_input_blob),
            get_blob_frame_seq(task_input_blob), get_blob_cli_id(task_input_blob), sys_now_in_sec()-start_time); 
   fclose(log);
#endif*/
   copy_blob_meta(temp, task_input_blob);
   overlapped_tile_data** temp_region_and_data = adjacent_reuse_data_deserialization(model, get_blob_task_id(temp), (float*)temp->data, get_blob_frame_seq(temp), reuse_data_is_required);
   place_adjacent_deserialized_data(model, get_blob_task_id(temp), temp_region_and_data, reuse_data_is_required);
   free_blob(temp);

   close_service_connection(conn);
}
#endif

static inline void process_task(device_ctxt* ctxt, blob* temp, bool is_reuse){
   cnn_model* model = (cnn_model*)(ctxt->model);
   blob* result;
   set_model_input(model, (float*)temp->data);
   forward_partition(model, get_blob_task_id(temp), is_reuse);  
#if DATA_REUSE
   set_coverage(model->ftp_para_reuse, get_blob_task_id(temp), get_blob_frame_seq(temp));
   send_reuse_data(ctxt, temp);
#endif
   result = new_blob_and_copy_data(0, 
                                      get_model_byte_size(model, model->ftp_para->fused_layers-1), 
                                      (uint8_t*)(get_model_output(model, model->ftp_para->fused_layers-1))
                                     );
   copy_blob_meta(result, temp);
   enqueue(ctxt->result_queue, result); 
   free_blob(result);
#if DEBUG_LOG
   char logname[20];
   sprintf(logname, "edge_%d.log", ctxt->this_cli_id);
   FILE *log = fopen(logname, "a");
   fprintf(log, "Task %d, from cli %d, frame is %d, time: %f\n", 
           get_blob_task_id(temp), get_blob_cli_id(temp), get_blob_frame_seq(temp), sys_now_in_sec()-start_time);
   fclose(log);
#endif
}


void partition_frame_and_perform_inference_thread(void *arg){
   device_ctxt* ctxt = (device_ctxt*)arg;
   cnn_model* model = (cnn_model*)(ctxt->model);
#ifdef NNPACK
   nnp_initialize();
   model->net->threadpool = pthreadpool_create(THREAD_NUM);
#endif
   blob* temp;
   uint32_t frame_num;
   bool* reuse_data_is_required;   
   for(frame_num = 0; frame_num < FRAME_NUM; frame_num ++){
      /*Wait for i/o device input*/
      /*recv_img()*/

      /*Load image and partition, fill task queues*/
      load_image_as_model_input(model, frame_num);
      partition_and_enqueue(ctxt, frame_num);
      register_client(ctxt);

      /*Dequeue and process task*/
      while(1){
         temp = try_dequeue(ctxt->task_queue);
         if(temp == NULL) break;
         bool data_ready = false;
#if DEBUG_DEEP_EDGE
         printf("====================Processing task id is %d, data source is %d, frame_seq is %d====================\n", get_blob_task_id(temp), get_blob_cli_id(temp), get_blob_frame_seq(temp));
#endif/*DEBUG_DEEP_EDGE*/
#if DATA_REUSE
         data_ready = is_reuse_ready(model->ftp_para_reuse, get_blob_task_id(temp), frame_num);
         if((model->ftp_para_reuse->schedule[get_blob_task_id(temp)] == 1) && data_ready) {
            blob* shrinked_temp = new_blob_and_copy_data(get_blob_task_id(temp), 
                       (model->ftp_para_reuse->shrinked_input_size[get_blob_task_id(temp)]),
                       (uint8_t*)(model->ftp_para_reuse->shrinked_input[get_blob_task_id(temp)]));
            copy_blob_meta(shrinked_temp, temp);
            free_blob(temp);
            temp = shrinked_temp;


            reuse_data_is_required = check_missing_coverage(model, get_blob_task_id(temp), get_blob_frame_seq(temp));
#if DEBUG_DEEP_EDGE
            printf("Request data from gateway, is there anything missing locally? ...\n");
            print_reuse_data_is_required(reuse_data_is_required);
#endif/*DEBUG_DEEP_EDGE*/
            request_reuse_data(ctxt, temp, reuse_data_is_required);
            free(reuse_data_is_required);
         }
#if DEBUG_DEEP_EDGE
         if((model->ftp_para_reuse->schedule[get_blob_task_id(temp)] == 1) && (!data_ready))
            printf("The reuse data is not ready yet!\n");
#endif/*DEBUG_DEEP_EDGE*/

#endif/*DATA_REUSE*/
         process_task(ctxt, temp, data_ready);
         free_blob(temp);
#if DEBUG_COMMU_SIZE
         printf("======Communication size at edge is: %f======\n", ((double)commu_size)/(1024.0*1024.0*FRAME_NUM));
#endif
      }

      /*Unregister and prepare for next image*/
      cancel_client(ctxt);
   }
#ifdef NNPACK
   pthreadpool_destroy(model->net->threadpool);
   nnp_deinitialize();
#endif
}




void steal_partition_and_perform_inference_thread(void *arg){
   device_ctxt* ctxt = (device_ctxt*)arg;
   /*Check gateway for possible stealing victims*/
#ifdef NNPACK
   cnn_model* model = (cnn_model*)(ctxt->model);
   nnp_initialize();
   model->net->threadpool = pthreadpool_create(THREAD_NUM);
#endif
   service_conn* conn;
   blob* temp;
   while(1){
      conn = connect_service(TCP, ctxt->gateway_local_addr, WORK_STEAL_PORT);
      send_request("steal_gateway", 20, conn);
      temp = recv_data(conn);
      close_service_connection(conn);
      if(temp->id == -1){
         free_blob(temp);
         sys_sleep(100);
         continue;
      }
      
      conn = connect_service(TCP, (const char *)temp->data, WORK_STEAL_PORT);
      send_request("steal_client", 20, conn);
      free_blob(temp);
      temp = recv_data(conn);
      if(temp->id == -1){
         close_service_connection(conn);
         free_blob(temp);
         sys_sleep(100);
         continue;
      }
      bool data_ready = true;
#if DATA_REUSE
      blob* reuse_info_blob = recv_data(conn);
      bool* reuse_data_is_required = (bool*) reuse_info_blob->data;
      request_reuse_data(ctxt, temp, reuse_data_is_required);
      if(!need_reuse_data_from_gateway(reuse_data_is_required)) data_ready = false; 
#if DEBUG_DEEP_EDGE
      printf("====================Processing task id is %d, data source is %d, frame_seq is %d====================\n", get_blob_task_id(temp), get_blob_cli_id(temp), get_blob_frame_seq(temp));
      printf("Request data from gateway, is the reuse data ready? ...\n");
      print_reuse_data_is_required(reuse_data_is_required);
#endif

      free_blob(reuse_info_blob);
#endif
      close_service_connection(conn);
      process_task(ctxt, temp, data_ready);
      free_blob(temp);
   }
#ifdef NNPACK
   pthreadpool_destroy(model->net->threadpool);
   nnp_deinitialize();
#endif
}


/*defined in gateway.h from darkiot
void send_result_thread;
*/


/*Function handling steal reqeust*/
#if DATA_REUSE
void* steal_client_reuse_aware(void* srv_conn, void* arg){
   printf("steal_client_reuse_aware ... ... at time %f\n", sys_now_in_sec()-start_time);
   service_conn *conn = (service_conn *)srv_conn;
   device_ctxt* ctxt = (device_ctxt*)arg;
   cnn_model* edge_model = (cnn_model*)(ctxt->model);

   blob* temp = try_dequeue(ctxt->task_queue);
   if(temp == NULL){
      char data[20]="empty";
      temp = new_blob_and_copy_data(-1, 20, (uint8_t*)data);
      send_data(temp, conn);
      free_blob(temp);
      return NULL;
   }
#if DEBUG_DEEP_EDGE
   printf("Stolen local task is %d\n", temp->id);
#endif

   uint32_t task_id = get_blob_task_id(temp);
   bool* reuse_data_is_required = (bool*)malloc(sizeof(bool)*4);
   uint32_t position;
   for(position = 0; position < 4; position++){
      reuse_data_is_required[position] = false;
   }

   if(edge_model->ftp_para_reuse->schedule[get_blob_task_id(temp)] == 1 && is_reuse_ready(edge_model->ftp_para_reuse, get_blob_task_id(temp), get_blob_frame_seq(temp))) {
      uint32_t position;
      int32_t* adjacent_id = get_adjacent_task_id_list(edge_model, task_id);
      for(position = 0; position < 4; position++){
         if(adjacent_id[position]==-1) continue;
         reuse_data_is_required[position] = true;
      }
      free(adjacent_id);
      blob* shrinked_temp = new_blob_and_copy_data(get_blob_task_id(temp), 
                       (edge_model->ftp_para_reuse->shrinked_input_size[get_blob_task_id(temp)]),
                       (uint8_t*)(edge_model->ftp_para_reuse->shrinked_input[get_blob_task_id(temp)]));
      copy_blob_meta(shrinked_temp, temp);
      free_blob(temp);
      temp = shrinked_temp;
   }
   send_data(temp, conn);
#if DEBUG_COMMU_SIZE
   commu_size = commu_size + temp->size;
#endif
   free_blob(temp);

   /*Send bool variables for different positions*/
   temp = new_blob_and_copy_data(task_id, 
                       sizeof(bool)*4,
                       (uint8_t*)(reuse_data_is_required));
   free(reuse_data_is_required);
   send_data(temp, conn);
   free_blob(temp);

   return NULL;
}

void* update_coverage(void* srv_conn, void* arg){
   printf("update_coverage ... ... \n");
   service_conn *conn = (service_conn *)srv_conn;
   device_ctxt* ctxt = (device_ctxt*)arg;
   cnn_model* edge_model = (cnn_model*)(ctxt->model);

   blob* temp = recv_data(conn);
#if DEBUG_DEEP_EDGE
   printf("set coverage for task %d\n", get_blob_task_id(temp));
#endif
   set_coverage(edge_model->ftp_para_reuse, get_blob_task_id(temp), get_blob_frame_seq(temp));
   set_missing(edge_model->ftp_para_reuse, get_blob_task_id(temp), get_blob_frame_seq(temp));
   free_blob(temp);
   return NULL;
}
#endif

void deepthings_serve_stealing_thread(void *arg){
#if DATA_REUSE
   const char* request_types[]={"steal_client", "update_coverage"};
   void* (*handlers[])(void*, void*) = {steal_client_reuse_aware, update_coverage};
#else
   const char* request_types[]={"steal_client"};
   void* (*handlers[])(void*, void*) = {steal_client};
#endif
   int wst_service = service_init(WORK_STEAL_PORT, TCP);
#if DATA_REUSE
   start_service(wst_service, TCP, request_types, 2, handlers, arg);
#else
   start_service(wst_service, TCP, request_types, 1, handlers, arg);
#endif
   close_service(wst_service);
}


void deepthings_stealer_edge(uint32_t N, uint32_t M, uint32_t fused_layers, char* network, char* weights, uint32_t edge_id){
   device_ctxt* ctxt = deepthings_edge_init(N, M, fused_layers, network, weights, edge_id);
   exec_barrier(START_CTRL, TCP, ctxt);
#if DEBUG_TIMING
   start_time = sys_now_in_sec();
#endif
   sys_thread_t t1 = sys_thread_new("steal_partition_and_perform_inference_thread", steal_partition_and_perform_inference_thread, ctxt, 0, 0);
   sys_thread_t t2 = sys_thread_new("send_result_thread", send_result_thread, ctxt, 0, 0);

   sys_thread_join(t1);
   sys_thread_join(t2);
}

void deepthings_victim_edge(uint32_t N, uint32_t M, uint32_t fused_layers, char* network, char* weights, uint32_t edge_id){
   device_ctxt* ctxt = deepthings_edge_init(N, M, fused_layers, network, weights, edge_id);
   exec_barrier(START_CTRL, TCP, ctxt);
#if DEBUG_TIMING
   start_time = sys_now_in_sec();
#endif
   sys_thread_t t1 = sys_thread_new("partition_frame_and_perform_inference_thread", partition_frame_and_perform_inference_thread, ctxt, 0, 0);
   sys_thread_t t2 = sys_thread_new("send_result_thread", send_result_thread, ctxt, 0, 0);
   sys_thread_t t3 = sys_thread_new("deepthings_serve_stealing_thread", deepthings_serve_stealing_thread, ctxt, 0, 0);

   sys_thread_join(t1);
   sys_thread_join(t2);
   sys_thread_join(t3);
}

