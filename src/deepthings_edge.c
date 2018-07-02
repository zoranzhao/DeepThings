#include "deepthings_edge.h"
#include "config.h"
#include "ftp.h"
#include "inference_engine_helper.h"
#include "frame_partitioner.h"
#include "reuse_data_serialization.h"
static cnn_model* edge_model;

cnn_model* deepthings_edge_init(){
   init_client();
   cnn_model* model;
   model = load_cnn_model((char*)"models/yolo.cfg", (char*)"models/yolo.weights");
   model->ftp_para = preform_ftp(PARTITIONS_H, PARTITIONS_W, FUSED_LAYERS, model->net_para);
#if DATA_REUSE
   model->ftp_para_reuse = preform_ftp_reuse(model->net_para, model->ftp_para);
#endif
   return model;
}

#if DATA_REUSE
void send_reuse_data(cnn_model* model, blob* task_input_blob){
   /*if task doesn't generate any reuse_data*/
   if(model->ftp_para_reuse->schedule[get_blob_task_id(task_input_blob)] == 1) return;

   service_conn* conn;

   blob* temp  = self_reuse_data_serialization(model, get_blob_task_id(task_input_blob), get_blob_frame_seq(task_input_blob));
   conn = connect_service(TCP, GATEWAY, WORK_STEAL_PORT);
   send_request("reuse_data", 20, conn);
#if DEBUG_DEEP_EDGE
   printf("send self reuse data for task %d:%d \n", get_blob_cli_id(task_input_blob), get_blob_task_id(task_input_blob)); 
#endif
   copy_blob_meta(temp, task_input_blob);
   send_data(temp, conn);
   free_blob(temp);
   close_service_connection(conn);

}

void request_reuse_data_in_data_source(cnn_model* model, blob* task_input_blob){
   /*if task doesn't require any reuse_data*/
   if(model->ftp_para_reuse->schedule[get_blob_task_id(task_input_blob)] == 0) return;

   service_conn* conn;
   conn = connect_service(TCP, GATEWAY, WORK_STEAL_PORT);
   send_request("request_reuse_data", 20, conn);
   char data[20]="empty";
   blob* temp = new_blob_and_copy_data(get_blob_task_id(task_input_blob), 20, (uint8_t*)data);
   copy_blob_meta(temp, task_input_blob);
   send_data(temp, conn);
   free_blob(temp);
#if DEBUG_DEEP_EDGE
   printf("Request reuse data for task %d:%d \n", get_blob_cli_id(task_input_blob), get_blob_task_id(task_input_blob)); 
#endif

   bool* reuse_data_is_required = check_local_coverage(model, get_blob_task_id(temp), get_blob_frame_seq(temp));
   temp = new_blob_and_copy_data(get_blob_task_id(task_input_blob), sizeof(bool)*4, (uint8_t*)reuse_data_is_required);
   copy_blob_meta(temp, task_input_blob);
   send_data(temp, conn);
   free_blob(temp);

   if(need_reuse_data_from_gateway(reuse_data_is_required)){
      temp = recv_data(conn);
      copy_blob_meta(temp, task_input_blob);
      overlapped_tile_data** temp_region_and_data = adjacent_reuse_data_deserialization(model, get_blob_task_id(temp), (float*)temp->data, get_blob_frame_seq(temp), reuse_data_is_required);
      place_adjacent_deserialized_data(model, get_blob_task_id(temp), temp_region_and_data, reuse_data_is_required);
      free_blob(temp);
   }
   close_service_connection(conn);
   free(reuse_data_is_required);
}

void request_reuse_data_in_stealer(cnn_model* model, blob* task_input_blob, bool* reuse_data_is_required){
   /*if task doesn't require any reuse_data*/
   if(model->ftp_para_reuse->schedule[get_blob_task_id(task_input_blob)] == 0) return;

   service_conn* conn;
   conn = connect_service(TCP, GATEWAY, WORK_STEAL_PORT);
   send_request("request_reuse_data", 20, conn);
   char data[20]="empty";
   blob* temp = new_blob_and_copy_data(get_blob_task_id(task_input_blob), 20, (uint8_t*)data);
   copy_blob_meta(temp, task_input_blob);
   send_data(temp, conn);
   free_blob(temp);
#if DEBUG_DEEP_EDGE
   printf("Request reuse data for task %d:%d \n", get_blob_cli_id(task_input_blob), get_blob_task_id(task_input_blob)); 
#endif

   temp = new_blob_and_copy_data(get_blob_task_id(task_input_blob), sizeof(bool)*4, (uint8_t*)reuse_data_is_required);
   copy_blob_meta(temp, task_input_blob);
   send_data(temp, conn);
   free_blob(temp);

   if(need_reuse_data_from_gateway(reuse_data_is_required)){
      temp = recv_data(conn);
      copy_blob_meta(temp, task_input_blob);
      overlapped_tile_data** temp_region_and_data = adjacent_reuse_data_deserialization(model, get_blob_task_id(temp), (float*)temp->data, get_blob_frame_seq(temp), reuse_data_is_required);
      place_adjacent_deserialized_data(model, get_blob_task_id(temp), temp_region_and_data, reuse_data_is_required);
      free_blob(temp);
   }
   close_service_connection(conn);
}
#endif

static inline void process_task(cnn_model* model, blob* temp){
   blob* result;
   set_model_input(model, (float*)temp->data);
   forward_partition(model, get_blob_task_id(temp));  
   result = new_blob_and_copy_data(0, 
                                      get_model_byte_size(model, model->ftp_para->fused_layers-1), 
                                      (uint8_t*)(get_model_output(model, model->ftp_para->fused_layers-1))
                                     );
#if DATA_REUSE
   send_reuse_data(model, temp);
#endif
   copy_blob_meta(result, temp);
   enqueue(result_queue, result); 
   free_blob(result);

}


void partition_frame_and_perform_inference_thread(void *arg){
   cnn_model* model = (cnn_model*)arg;
#ifdef NNPACK
   nnp_initialize();
   model->net->threadpool = pthreadpool_create(THREAD_NUM);
#endif
   blob* temp;
   uint32_t frame_num;
   for(frame_num = 0; frame_num < FRAME_NUM; frame_num ++){
      /*Wait for i/o device input*/
      /*recv_img()*/

      /*Load image and partition, fill task queues*/
      load_image_as_model_input(model, frame_num);
      partition_and_enqueue(model, frame_num);
      register_client();

      /*Dequeue and process task*/
      while(1){
         temp = try_dequeue(task_queue);
         if(temp == NULL) break;
         printf("Task id is %d\n", temp->id);
#if DATA_REUSE
/*
&& is_reuse_ready(model->ftp_para_reuse, get_blob_task_id(temp))
*/
         if(model->ftp_para_reuse->schedule[get_blob_task_id(temp)] == 1) {
            blob* shrinked_temp = new_blob_and_copy_data(get_blob_task_id(temp), 
                       (model->ftp_para_reuse->shrinked_input_size[get_blob_task_id(temp)]),
                       (uint8_t*)(model->ftp_para_reuse->shrinked_input[get_blob_task_id(temp)]));
            copy_blob_meta(shrinked_temp, temp);
            free_blob(temp);
            temp = shrinked_temp;
         }
         request_reuse_data_in_data_source(model, temp);
#endif
         process_task(model, temp);
         free_blob(temp);
      }

      /*Unregister and prepare for next image*/
      cancel_client();
   }
#ifdef NNPACK
   pthreadpool_destroy(model->net->threadpool);
   nnp_deinitialize();
#endif
}




void steal_partition_and_perform_inference_thread(void *arg){
   cnn_model* model = (cnn_model*)arg;
   /*Check gateway for possible stealing victims*/
#ifdef NNPACK
   nnp_initialize();
   model->net->threadpool = pthreadpool_create(THREAD_NUM);
#endif
   service_conn* conn;
   blob* temp;
   while(1){
      conn = connect_service(TCP, GATEWAY, WORK_STEAL_PORT);
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
         free_blob(temp);
         sys_sleep(100);
         continue;
      }
#if DATA_REUSE
      blob* reuse_info_blob = recv_data(conn);
      bool* reuse_data_is_required = (bool*) reuse_info_blob->data;
      request_reuse_data_in_stealer(model, temp, reuse_data_is_required);
      free_blob(reuse_info_blob);
#endif
      close_service_connection(conn);
      process_task(model, temp);
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

#if DATA_REUSE
void* steal_client_reuse_aware(void* srv_conn){
   printf("steal_client_reuse_aware ... ... \n");
   service_conn *conn = (service_conn *)srv_conn;
   blob* temp = try_dequeue(task_queue);
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

   if(edge_model->ftp_para_reuse->schedule[get_blob_task_id(temp)] == 1) {

      uint32_t position;
      int32_t adjacent_id[4];
      for(position = 0; position < 4; position++){
         adjacent_id[position]=-1;
      }
      ftp_parameters_reuse* ftp_para_reuse = edge_model->ftp_para_reuse;
      uint32_t j = task_id%(ftp_para_reuse->partitions_w);
      uint32_t i = task_id/(ftp_para_reuse->partitions_w);
      if((i+1)<(ftp_para_reuse->partitions_h)) adjacent_id[0] = ftp_para_reuse->task_id[i+1][j];
      /*get the left overlapped data from tile on the right*/
      if((j+1)<(ftp_para_reuse->partitions_w)) adjacent_id[1] = ftp_para_reuse->task_id[i][j+1];
      /*get the bottom overlapped data from tile above*/
      if(i>0) adjacent_id[2] = ftp_para_reuse->task_id[i-1][j];
      /*get the right overlapped data from tile on the left*/
      if(j>0) adjacent_id[3] = ftp_para_reuse->task_id[i][j-1];

      for(position = 0; position < 4; position++){
         if(adjacent_id[position]==-1) continue;
         reuse_data_is_required[position] = true;
      }

      blob* shrinked_temp = new_blob_and_copy_data(get_blob_task_id(temp), 
                       (edge_model->ftp_para_reuse->shrinked_input_size[get_blob_task_id(temp)]),
                       (uint8_t*)(edge_model->ftp_para_reuse->shrinked_input[get_blob_task_id(temp)]));
      copy_blob_meta(shrinked_temp, temp);
      free_blob(temp);
      temp = shrinked_temp;
   }
   send_data(temp, conn);
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
#endif

void deepthings_serve_stealing_thread(void *arg){
   const char* request_types[]={"steal_client"};
#if DATA_REUSE
   void* (*handlers[])(void*) = {steal_client_reuse_aware};
#else
   void* (*handlers[])(void*) = {steal_client};
#endif
   int wst_service = service_init(WORK_STEAL_PORT, TCP);
   start_service(wst_service, TCP, request_types, 1, handlers);
   close_service(wst_service);
}


void deepthings_stealer_edge(){


   cnn_model* model = deepthings_edge_init();
   exec_barrier(START_CTRL, TCP);

   sys_thread_t t1 = sys_thread_new("steal_partition_and_perform_inference_thread", steal_partition_and_perform_inference_thread, model, 0, 0);
   sys_thread_t t2 = sys_thread_new("send_result_thread", send_result_thread, model, 0, 0);

   sys_thread_join(t1);
   sys_thread_join(t2);

}

void deepthings_victim_edge(){


   cnn_model* model = deepthings_edge_init();
   edge_model = model;
   exec_barrier(START_CTRL, TCP);

   sys_thread_t t1 = sys_thread_new("partition_frame_and_perform_inference_thread", partition_frame_and_perform_inference_thread, model, 0, 0);
   sys_thread_t t2 = sys_thread_new("send_result_thread", send_result_thread, model, 0, 0);
   sys_thread_t t3 = sys_thread_new("deepthings_serve_stealing_thread", deepthings_serve_stealing_thread, model, 0, 0);

   sys_thread_join(t1);
   sys_thread_join(t2);
   sys_thread_join(t3);


}

