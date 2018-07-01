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
#endif
   return model;
}

static inline void process_task(cnn_model* model, blob* temp){
   blob* result;
   set_model_input(model, (float*)temp->data);
   forward_partition(model, get_blob_task_id(temp));  
   result = new_blob_and_copy_data(0, 
                                      get_model_byte_size(model, model->ftp_para->fused_layers-1), 
                                      (uint8_t*)(get_model_output(model, model->ftp_para->fused_layers-1))
                                     );
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
         if(model->ftp_para_reuse->schedule[get_blob_task_id(temp)] == 1 && is_reuse_ready(model->ftp_para_reuse, get_blob_task_id(temp))) {
             set_model_input(model, (float*)(model->ftp_para_reuse->shrinked_input[get_blob_task_id(temp)]));
         }
         else
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
void serve_stealing_thread;
*/
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
   sys_thread_t t3 = sys_thread_new("serve_stealing_thread", serve_stealing_thread, model, 0, 0);

   sys_thread_join(t1);
   sys_thread_join(t2);
   sys_thread_join(t3);


}

