#include "darkiot.h"
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

void* serve_steal(void* conn, void* arg){
   printf("serve_steal ... ... \n");
   blob* temp = recv_data((service_conn *)conn);
   send_data(temp, conn);
   write_blob_to_file("out1.jpg", temp);
   free_blob(temp);
   return NULL;
}

void* serve_result(void* conn, void* arg){
   printf("serve_result ... ... \n");
   blob* temp = recv_data((service_conn *)conn);
   send_data(temp, conn);
   write_blob_to_file("out2.jpg", temp);
   free_blob(temp);
   return NULL;
}

void* serve_sync(void* conn, void* arg){
   printf("serve_sync ... ... \n");
   blob* temp = recv_data((service_conn *)conn);
   send_data(temp, conn);
   write_blob_to_file("out3.jpg", temp);
   free_blob(temp);
   return NULL;
}

void server_thread(void *arg){
   const char* request_types[]={"steal", "result", "sync"};
   void* (*handlers[])(void*, void*) = {serve_steal, serve_result, serve_sync};

   int mapreduce_service = service_init(8080, TCP);
   printf("Service number is %d\n", mapreduce_service);

   start_service_for_n_times(mapreduce_service, TCP, request_types, 3, handlers, NULL, 1);
   printf("Service 1 is returned \n");
   start_service_for_n_times(mapreduce_service, TCP, request_types, 3, handlers, NULL, 1);
   printf("Service 2 is returned \n");
   start_service_for_n_times(mapreduce_service, TCP, request_types, 3, handlers, NULL, 1);
   printf("Service 3 is returned \n");

/*
   start_service(mapreduce_service, TCP, request_types, 3, handlers);
*/   
   start_service_for_n_times(mapreduce_service, TCP, request_types, 3, handlers, NULL, 3);
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
   char str[40]="Abracadabra, blablaboom";
   sys_thread_t t3 = sys_thread_new("work_stealing_thread", work_stealing_thread, str, 0, 0);
   sys_thread_t t1 = sys_thread_new("collect_result_thread", collect_result_thread, str, 0, 0);
   sys_thread_t t2 = sys_thread_new("merge_result_thread", merge_result_thread, str, 0, 0);
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
   test_wst(argc, argv);
   return 0;
}

