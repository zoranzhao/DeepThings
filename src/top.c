#include "darkiot.h"
#include "config.h"
#include "configure.h"
#include "ftp.h"
#include "inference_engine_helper.h"
#include "frame_partitioner.h"
#include <string.h>

static thread_safe_queue* q; 

void consumer(void *arg){
   blob* tmp;
   while(1){
      tmp = dequeue(q);
      printf("ID: %d, Data: %s\n", tmp->id, tmp->data);
      printf("number_of_node %d\n", q->number_of_node);
      free_blob(tmp);
   }
}

void producer(void *arg){
   char testdata1[30]="testdata1!";
   char testdata2[30]="testdata2!";
   char testdata3[30]="testdata3!";
   char testdata4[30]="testdata4!";
   blob* b1 = new_blob_and_copy_data(1, 30, (uint8_t*)testdata1);
   blob* b2 = new_blob_and_move_data(2, 30, (uint8_t*)testdata2);
   blob* b3 = new_blob_and_move_data(3, 30, (uint8_t*)testdata3);
   blob* b4 = new_blob_and_copy_data(4, 30, (uint8_t*)testdata4);
   blob* b5 = new_blob_and_alloc_data(5, 30);
   strcpy((char*)b5->data, "testdata5!");

   enqueue(q, b1);
   free_blob(b1);
   enqueue(q, b2);
   free_blob(b2);
   enqueue(q, b3);
   free_blob(b3);
   enqueue(q, b4);
   free_blob(b4);
   enqueue(q, b5);
   free_blob(b5);
}

void* serve_steal(void* conn){
   printf("serve_steal ... ... \n");
   blob* temp = recv_data((service_conn *)conn);
   send_data(temp, conn);
   write_blob_to_file("out1.jpg", temp);
   free_blob(temp);
   return NULL;
}

void* serve_result(void* conn){
   printf("serve_result ... ... \n");
   blob* temp = recv_data((service_conn *)conn);
   send_data(temp, conn);
   write_blob_to_file("out2.jpg", temp);
   free_blob(temp);
   return NULL;
}

void* serve_sync(void* conn){
   printf("serve_sync ... ... \n");
   blob* temp = recv_data((service_conn *)conn);
   send_data(temp, conn);
   write_blob_to_file("out3.jpg", temp);
   free_blob(temp);
   return NULL;
}

void server_thread(void *arg){
   const char* request_types[]={"steal", "result", "sync"};
   void* (*handlers[])(void*) = {serve_steal, serve_result, serve_sync};

   int mapreduce_service = service_init(8080, TCP);
   printf("Service number is %d\n", mapreduce_service);

   start_service_for_n_times(mapreduce_service, TCP, request_types, 3, handlers, 1);
   printf("Service 1 is returned \n");
   start_service_for_n_times(mapreduce_service, TCP, request_types, 3, handlers, 1);
   printf("Service 2 is returned \n");
   start_service_for_n_times(mapreduce_service, TCP, request_types, 3, handlers, 1);
   printf("Service 3 is returned \n");

/*
   start_service(mapreduce_service, TCP, request_types, 3, handlers);
*/   
   start_service_for_n_times(mapreduce_service, TCP, request_types, 3, handlers, 3);
   close_service(mapreduce_service);
}

void client_thread(void *arg){
   blob* temp = write_file_to_blob("test.jpg");
   char request1[20] = "steal";
   char request2[20] = "result";
   char request3[20] = "sync";

   service_conn* conn;
   blob* recv_temp;

   conn = connect_service(TCP, "10.145.80.46", 8080);
   send_request(request1, 20, conn);
   send_data(temp, conn);
   recv_temp = recv_data(conn);
   write_blob_to_file("out4.jpg", recv_temp);
   free_blob(recv_temp);
   close_service_connection(conn);

   conn = connect_service(TCP, "10.145.80.46", 8080);
   send_request(request2, 20, conn);
   send_data(temp, conn);
   recv_temp = recv_data(conn);
   write_blob_to_file("out5.jpg", recv_temp);
   free_blob(recv_temp);
   close_service_connection(conn);

   conn = connect_service(TCP, "10.145.80.46", 8080);
   send_request(request3, 20, conn);
   send_data(temp, conn);
   recv_temp = recv_data(conn);
   write_blob_to_file("out6.jpg", recv_temp);
   free_blob(recv_temp);
   close_service_connection(conn);

   conn = connect_service(TCP, "10.145.80.46", 8080);
   send_request(request1, 20, conn);
   send_data(temp, conn);
   recv_temp = recv_data(conn);
   write_blob_to_file("out7.jpg", recv_temp);
   free_blob(recv_temp);
   close_service_connection(conn);

   conn = connect_service(TCP, "10.145.80.46", 8080);
   send_request(request2, 20, conn);
   send_data(temp, conn);
   recv_temp = recv_data(conn);
   write_blob_to_file("out8.jpg", recv_temp);
   free_blob(recv_temp);
   close_service_connection(conn);

   conn = connect_service(TCP, "10.145.80.46", 8080);
   send_request(request3, 20, conn);
   send_data(temp, conn);
   recv_temp = recv_data(conn);
   write_blob_to_file("out9.jpg", recv_temp);
   free_blob(recv_temp);
   close_service_connection(conn);

   free_blob(temp);
}

void test_queue_remove(int argc, char **argv){
   q = new_queue(20); 
   producer(NULL);
   remove_by_id(q, 1);
   print_queue_by_id(q);
   remove_by_id(q, 2);
   print_queue_by_id(q);
   remove_by_id(q, 3);
   print_queue_by_id(q);
   remove_by_id(q, 4);
   print_queue_by_id(q);
   remove_by_id(q, 5);
   print_queue_by_id(q);
}

void test_queue(int argc, char **argv){
   q = new_queue(20); 
   sys_thread_t t1 = sys_thread_new("consumer", consumer, NULL, 0, 0);
   sys_thread_t t2 = sys_thread_new("producer", producer, NULL, 0, 0);
   sys_thread_join(t1);
   sys_thread_join(t2);
}

void test_network_api(int argc, char **argv){
   q = new_queue(20); 
   sys_thread_t t1 = sys_thread_new("server", server_thread, NULL, 0, 0);
   sys_thread_t t2 = sys_thread_new("client", client_thread, NULL, 0, 0);
   sys_thread_join(t1);
   sys_thread_join(t2);
   
}

/*
make ARGS="2 start" test
make ARGS="2 wst gateway" test
make ARGS="0 wst_s data_source" test
make ARGS="1 wst non_data_source" test
make ARGS="2 wst non_data_source" test
*/
void test_gateway_ctrl(){
   exec_barrier(START_CTRL, TCP);
}

void test_edge_ctrl(){
   exec_barrier(START_CTRL, TCP);
}

void test_gateway(){

   init_gateway();
   sys_thread_t t3 = sys_thread_new("work_stealing_thread", work_stealing_thread, NULL, 0, 0);
   sys_thread_t t1 = sys_thread_new("collect_result_thread", collect_result_thread, NULL, 0, 0);
   sys_thread_t t2 = sys_thread_new("merge_result_thread", merge_result_thread, NULL, 0, 0);
   exec_barrier(START_CTRL, TCP);

   sys_thread_join(t1);
   sys_thread_join(t2);
   sys_thread_join(t3);


}

void test_stealer_client(){
   exec_barrier(START_CTRL, TCP);
   init_client();
   sys_thread_t t1 = sys_thread_new("steal_and_process_thread", steal_and_process_thread, NULL, 0, 0);
   sys_thread_t t2 = sys_thread_new("send_result_thread", send_result_thread, NULL, 0, 0);

   sys_thread_join(t1);
   sys_thread_join(t2);

}

void test_victim_client(){
   exec_barrier(START_CTRL, TCP);
   init_client();
   sys_thread_t t1 = sys_thread_new("generate_and_process_thread", generate_and_process_thread, NULL, 0, 0);
   sys_thread_t t2 = sys_thread_new("send_result_thread", send_result_thread, NULL, 0, 0);
   sys_thread_t t3 = sys_thread_new("serve_stealing_thread", serve_stealing_thread, NULL, 0, 0);

   sys_thread_join(t1);
   sys_thread_join(t2);
   sys_thread_join(t3);


}

void test_wst(int argc, char **argv)
{
   printf("total_cli_num %d\n", atoi(argv[1]));
   printf("this_cli_id %d\n", atoi(argv[1]));

   this_cli_id = atoi(argv[1]);
   total_cli_num = atoi(argv[1]);
      
   if(0 == strcmp(argv[2], "start")){  
      printf("start\n");
      exec_start_gateway(START_CTRL, TCP);
   }else if(0 == strcmp(argv[2], "wst")){
      printf("Work stealing\n");
      if(0 == strcmp(argv[3], "non_data_source")){
         printf("non_data_source\n");
         test_stealer_client();
      }else if(0 == strcmp(argv[3], "data_source")){
         printf("data_source\n");
         test_victim_client();
      }else if(0 == strcmp(argv[3], "gateway")){
         printf("gateway\n");
         test_gateway();
      }
   }else if(0 == strcmp(argv[2], "wst_s")){
      printf("Work stealing with scheduling\n");
      if(0 == strcmp(argv[3], "non_data_source")){
         printf("non_data_source\n");
         test_stealer_client();
      }else if(0 == strcmp(argv[3], "data_source")){
         printf("data_source\n");
         test_victim_client();
      }else if(0 == strcmp(argv[3], "gateway")){
         printf("gateway\n");
         test_gateway();
      }
   }
}

int main(int argc, char **argv){
   /*test_queue_remove(argc, argv);*/
   /*test_wst(argc, argv);*/

   this_cli_id = 0;
   total_cli_num = 1;
   init_queues(total_cli_num);

   cnn_model* model = load_cnn_model((char*)"models/yolo.cfg", (char*)"models/yolo.weights");
   model->ftp_para = preform_ftp(5, 5, 16, model->net_para);
   model->ftp_para_reuse = preform_ftp_reuse(model->net_para, model->ftp_para);

/*
   for(int i = 0; i < 2; i++){
      for(int j = 0; j < 2; j++){
         printf("------------------(%3d,%3d)----------------\n", i, j);
         for(int l = 2; l >= 0; l--)
            print_tile_region(model->ftp_para->output_tiles[model->ftp_para->task_id[i][j]][l]);
         print_tile_region(model->ftp_para->input_tiles[model->ftp_para->task_id[i][j]][0]);
      }
   }
*/

   for(int frame_seq = 0; frame_seq < 4; frame_seq++){
      image_holder img = load_image_as_model_input(model, frame_seq);
      partition_and_enqueue(model, frame_seq);
      blob* temp;
      blob* result;
      while(1){
         temp = try_dequeue(task_queue);
         if(temp == NULL) break;
         set_model_input(model, (float*)temp->data);
         forward_partition(model, get_blob_task_id(temp));      
         result = new_blob_and_copy_data(0, 
                                      get_model_byte_size(model, model->ftp_para->fused_layers-1), 
                                      (uint8_t*)(get_model_output(model, model->ftp_para->fused_layers-1))
                                     );
         copy_blob_meta(result, temp);
         enqueue(results_pool[this_cli_id], result);
         /*enqueue(result_queue, result);*/ 
         free_blob(result);
         free_blob(temp);
      }

      enqueue(ready_pool, new_empty_blob(this_cli_id));

      float* fused_output = dequeue_and_merge(model);
      set_model_input(model, fused_output);
      forward_all(model, model->ftp_para->fused_layers);   
      free(fused_output);
      draw_object_boxes(model, frame_seq);
      free_image_holder(model, img);
   }
/*
   for(int frame_seq = 0; frame_seq < 4; frame_seq++){
      image_holder img = load_image_as_model_input(model, frame_seq);
      forward_all(model, 0);   
      draw_object_boxes(model, frame_seq);
      free_image_holder(model, img);
   }
*/


   return 0;
}

