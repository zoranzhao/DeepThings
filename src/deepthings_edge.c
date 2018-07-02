#include "deepthings_edge.h"
#include "config.h"
#include "ftp.h"
#include "inference_engine_helper.h"
#include "frame_partitioner.h"
#include "reuse_data_serialization.h"

cnn_model* deepthings_edge_init(){
   init_client();
   cnn_model* model;
   model = load_cnn_model((char*)"models/yolo.cfg", (char*)"models/yolo.weights");
   model->ftp_para = preform_ftp(PARTITIONS_H, PARTITIONS_W, FUSED_LAYERS, model->net_para);
#if DATA_REUSE
   model->ftp_para_reuse = preform_ftp_reuse(model->net_para, model->ftp_para);
   shrinked_task_queue = new_queue(MAX_QUEUE_SIZE);
   schedule_task_queue = new_queue(MAX_QUEUE_SIZE);
#endif
   return model;
}

#if DATA_REUSE
void send_reuse_data(cnn_model* model, blob* task_input_blob){
   /*if task doesn't generate any reuse_data*/
   if(model->ftp_para_reuse->schedule[get_blob_task_id(task_input_blob)] == 1) return;

   service_conn* conn;
   bool tmp[4];
   tmp[0] =true;
   tmp[1] =true;
   tmp[2] =true;
   tmp[3] =true;
   
   blob* temp = reuse_data_serialization(model, get_blob_cli_id(task_input_blob), get_blob_frame_seq(task_input_blob), tmp);
   conn = connect_service(TCP, GATEWAY, WORK_STEAL_PORT);
   send_request("reuse_data", 20, conn);
#if DEBUG_DEEP_EDGE
   printf("send reuse data for task %d:%d \n", get_blob_cli_id(task_input_blob), get_blob_task_id(task_input_blob)); 
#endif
   copy_blob_meta(temp, task_input_blob);
   send_data(temp, conn);
   free_blob(temp);
   close_service_connection(conn);
}


void request_reuse_data(cnn_model* model, blob* task_input_blob){
   /*if task doesn't require any reuse_data*/
   if(model->ftp_para_reuse->schedule[get_blob_task_id(task_input_blob)] == 0) return;

   service_conn* conn;
   bool tmp[4];
   tmp[0] =true;
   tmp[1] =true;
   tmp[2] =true;
   tmp[3] =true;
   
   char data[20]="empty";
   blob* temp = new_blob_and_copy_data(get_blob_task_id(task_input_blob), 20, (uint8_t*)data);
   copy_blob_meta(temp, task_input_blob);

   conn = connect_service(TCP, GATEWAY, WORK_STEAL_PORT);
   send_request("request_reuse_data", 20, conn);
   send_data(temp, conn);
   free_blob(temp);
#if DEBUG_DEEP_EDGE
   printf("Request reuse data for task %d:%d \n", get_blob_cli_id(task_input_blob), get_blob_task_id(task_input_blob)); 
#endif
   temp = recv_data(conn);
   copy_blob_meta(temp, task_input_blob);
   overlapped_tile_data** temp_region_and_data = reuse_data_deserialization(model, get_blob_task_id(temp), (float*)temp->data, get_blob_frame_seq(temp), tmp);
   place_deserialized_data(model, get_blob_task_id(temp), temp_region_and_data, tmp);
   free_blob(temp);
   close_service_connection(conn);
}
#endif

static inline void process_task(cnn_model* model, blob* temp){
#if DATA_REUSE
   request_reuse_data(model, temp);
#endif
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
         blob *schedule = dequeue(schedule_task_queue);
         if(get_blob_task_id(schedule)==1){
            free_blob(temp);
            temp = dequeue(shrinked_task_queue);
         }
         free_blob(schedule);
/*
         if(model->ftp_para_reuse->schedule[get_blob_task_id(temp)] == 1 && is_reuse_ready(model->ftp_para_reuse, get_blob_task_id(temp))) {
             set_model_input(model, (float*)(model->ftp_para_reuse->shrinked_input[get_blob_task_id(temp)]));
         }
         else
*/
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
      close_service_connection(conn);
      if(temp->id == -1){
         free_blob(temp);
         sys_sleep(100);
         continue;
      }
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

   blob* schedule = dequeue(schedule_task_queue);
   if(get_blob_task_id(schedule)==1){
#if DEBUG_DEEP_EDGE
      printf("Send reuse data!\n");
#endif
      free_blob(temp);
      temp = dequeue(shrinked_task_queue);
   }
   free_blob(schedule); 
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
   exec_barrier(START_CTRL, TCP);

   sys_thread_t t1 = sys_thread_new("partition_frame_and_perform_inference_thread", partition_frame_and_perform_inference_thread, model, 0, 0);
   sys_thread_t t2 = sys_thread_new("send_result_thread", send_result_thread, model, 0, 0);
   sys_thread_t t3 = sys_thread_new("deepthings_serve_stealing_thread", deepthings_serve_stealing_thread, model, 0, 0);

   sys_thread_join(t1);
   sys_thread_join(t2);
   sys_thread_join(t3);


}

