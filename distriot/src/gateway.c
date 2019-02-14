#include "gateway.h"

/*Allocated spaces for gateway devices*/
device_ctxt* init_gateway(uint32_t cli_num, const char** edge_addr_list){

   device_ctxt* ctxt = (device_ctxt*)malloc(sizeof(device_ctxt)); 
   uint32_t i;

/*Queues used in gateway device*/
   ctxt->results_pool = (thread_safe_queue**)malloc(sizeof(thread_safe_queue*)*cli_num);
   ctxt->results_counter = (uint32_t*)malloc(sizeof(uint32_t)*cli_num);
   for(i = 0; i < cli_num; i++){
      ctxt->results_pool[i] = new_queue(MAX_QUEUE_SIZE);
      ctxt->results_counter[i] = 0;
   } 
   ctxt->ready_pool = new_queue(MAX_QUEUE_SIZE); 
   ctxt->registration_list = new_queue(MAX_QUEUE_SIZE); 
   ctxt->total_cli_num = cli_num;
   ctxt->addr_list = (char**)malloc(sizeof(char*)*cli_num);
   for(i = 0; i < cli_num; i++){
      ctxt->addr_list[i] = (char*)malloc(sizeof(char)*ADDR_LEN);
      strcpy(ctxt->addr_list[i], edge_addr_list[i]);
   }
 


   return ctxt;
}

void* result_gateway(void* srv_conn, void* arg){
   printf("result_gateway ... ... \n");
   device_ctxt* ctxt = (device_ctxt*)arg;
   service_conn *conn = (service_conn *)srv_conn;
   int32_t cli_id;
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
#if DEBUG_FLAG
   printf("Result from %d: %s is for client %d, total number recved is %d\n", processing_cli_id, ip_addr, cli_id, ctxt->results_counter[cli_id]);
#endif
   enqueue(ctxt->results_pool[cli_id], temp);
   free_blob(temp);
   ctxt->results_counter[cli_id]++;
   if(ctxt->results_counter[cli_id] == ctxt->batch_size){
      temp = new_empty_blob(cli_id);
      enqueue(ctxt->ready_pool, temp);
      free_blob(temp);
      ctxt->results_counter[cli_id] = 0;
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
   device_ctxt* ctxt = (device_ctxt*)arg;
   blob* temp = dequeue(ctxt->ready_pool);
   int32_t cli_id = temp->id;
   free_blob(temp);
#if DEBUG_FLAG
   printf("Results for client %d are all collected\n", cli_id);
#endif
   uint32_t batch = 0;
   for(batch = 0; batch < ctxt->batch_size; batch ++){
      temp = dequeue(ctxt->results_pool[cli_id]);
      free_blob(temp);
   }
}

void* register_gateway(void* srv_conn, void *arg){
   printf("register_gateway ... ... \n");
   service_conn *conn = (service_conn *)srv_conn;
   device_ctxt* ctxt = (device_ctxt*)arg;
   char ip_addr[ADDRSTRLEN];
   get_dest_ip_string(ip_addr, conn);
   blob* temp = new_blob_and_copy_data(get_client_id(ip_addr, ctxt), ADDRSTRLEN, (uint8_t*)ip_addr);
   enqueue(ctxt->registration_list, temp);
   free_blob(temp);
#if DEBUG_FLAG
   queue_node* cur = ctxt->registration_list->head;
   if (ctxt->registration_list->head == NULL){
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
   device_ctxt* ctxt = (device_ctxt*)arg;
   char ip_addr[ADDRSTRLEN];
   get_dest_ip_string(ip_addr, conn);
   int32_t cli_id = get_client_id(ip_addr, ctxt);
   remove_by_id(ctxt->registration_list, cli_id);
   return NULL;
}

void* steal_gateway(void* srv_conn, void *arg){
   service_conn *conn = (service_conn *)srv_conn;
   device_ctxt* ctxt = (device_ctxt*)arg;
   blob* temp = try_dequeue(ctxt->registration_list);
   if(temp == NULL){
      char ip_addr[ADDRSTRLEN]="empty";
      temp = new_blob_and_copy_data(-1, ADDRSTRLEN, (uint8_t*)ip_addr);
   }else{
      enqueue(ctxt->registration_list, temp);
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

