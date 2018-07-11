#include "gateway.h"
#include "global_queues.h"

/*Allocated spaces for gateway devices*/
void init_gateway(){
   results_pool = (thread_safe_queue**)malloc(sizeof(thread_safe_queue*)*total_cli_num);
   results_counter = (uint32_t*)malloc(sizeof(uint32_t)*total_cli_num);
   uint32_t i;
   for(i = 0; i < total_cli_num; i++){
      results_pool[i] = new_queue(MAX_QUEUE_SIZE);
      results_counter[i] = 0;
   } 
   ready_pool = new_queue(MAX_QUEUE_SIZE); 
   registration_list = new_queue(MAX_QUEUE_SIZE); 
}

void* result_gateway(void* srv_conn, void* arg){
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
   if(results_counter[cli_id] == BATCH_SIZE){
      temp = new_empty_blob(cli_id);
      enqueue(ready_pool, temp);
      free_blob(temp);
      results_counter[cli_id] = 0;
   }

   return NULL;
}

void collect_result_thread(void *arg){
   const char* request_types[]={"result_gateway"};
   void* (*handlers[])(void*,void*) = {result_gateway};
   int result_service = service_init(RESULT_COLLECT_PORT, TCP);
   start_service(result_service, TCP, request_types, 1, handlers, arg);
   close_service(result_service);
}

void merge_result_thread(void *arg){
   blob* temp = dequeue(ready_pool);
   int32_t cli_id = temp->id;
   free_blob(temp);
#if DEBUG_FLAG
   printf("Results for client %d are all collected\n", cli_id);
#endif
   uint32_t batch = 0;
   for(batch = 0; batch < BATCH_SIZE; batch ++){
      temp = dequeue(results_pool[cli_id]);
      free_blob(temp);
   }
}

void* register_gateway(void* srv_conn, void *arg){
   printf("register_gateway ... ... \n");
   service_conn *conn = (service_conn *)srv_conn;
   char ip_addr[ADDRSTRLEN];
   inet_ntop(conn->serv_addr_ptr->sin_family, &(conn->serv_addr_ptr->sin_addr), ip_addr, ADDRSTRLEN);  
   blob* temp = new_blob_and_copy_data(get_client_id(ip_addr), ADDRSTRLEN, (uint8_t*)ip_addr);
   enqueue(registration_list, temp);
   free_blob(temp);
#if DEBUG_FLAG
   queue_node* cur = registration_list->head;
   if (registration_list->head == NULL){
      printf("No client is registered!\n");
   }
   while (1) {
      if (cur->next == NULL){
         printf("%d: %s,\n", cur->item->id, ((char*)(cur->item->data)));
         break;
      }
      printf("%d: %s\n", cur->item->id, ((char*)(cur->item->data)));
      cur = cur->next;
   } 
#endif
   return NULL;
}

void* cancel_gateway(void* srv_conn, void *arg){
   printf("cancel_gateway ... ... \n");
   service_conn *conn = (service_conn *)srv_conn;
   char ip_addr[ADDRSTRLEN];
   inet_ntop(conn->serv_addr_ptr->sin_family, &(conn->serv_addr_ptr->sin_addr), ip_addr, ADDRSTRLEN);  
   int32_t cli_id = get_client_id(ip_addr);
   remove_by_id(registration_list, cli_id);
   return NULL;
}

void* steal_gateway(void* srv_conn, void *arg){
   service_conn *conn = (service_conn *)srv_conn;
   blob* temp = try_dequeue(registration_list);
   if(temp == NULL){
      char ip_addr[ADDRSTRLEN]="empty";
      temp = new_blob_and_copy_data(-1, ADDRSTRLEN, (uint8_t*)ip_addr);
   }else{
      enqueue(registration_list, temp);
   }
   send_data(temp, conn);
   free_blob(temp);
   return NULL;
}

void work_stealing_thread(void *arg){
   const char* request_types[]={"register_gateway", "cancel_gateway", "steal_gateway"};
   void* (*handlers[])(void*,void*) = {register_gateway, cancel_gateway, steal_gateway};
   int wst_service = service_init(WORK_STEAL_PORT, TCP);
   start_service(wst_service, TCP, request_types, 3, handlers, arg);
   close_service(wst_service);
}

