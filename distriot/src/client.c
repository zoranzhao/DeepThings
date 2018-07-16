#include "client.h"

device_ctxt* init_client(uint32_t cli_id){
   device_ctxt* ctxt = (device_ctxt*)malloc(sizeof(device_ctxt)); 
/*Queues used in edge device*/
   ctxt->task_queue = new_queue(MAX_QUEUE_SIZE);
   ctxt->result_queue = new_queue(MAX_QUEUE_SIZE); 
   ctxt->this_cli_id = cli_id;

   return ctxt;
}


void register_client(device_ctxt* ctxt){
   char request_type[20] = "register_gateway";
   service_conn* conn = connect_service(TCP, ctxt->gateway_local_addr, WORK_STEAL_PORT);
   send_request(request_type, 20, conn);
   close_service_connection(conn);
}

void cancel_client(device_ctxt* ctxt){
   char request_type[20] = "cancel_gateway";
   service_conn* conn = connect_service(TCP, ctxt->gateway_local_addr, WORK_STEAL_PORT);
   send_request(request_type, 20, conn);
   close_service_connection(conn);
}

static void process_task(blob* temp, device_ctxt* ctxt){
   blob* result;
   char data[20] = "output_data";
   result = new_blob_and_copy_data(temp->id, 20, (uint8_t*)data);
   copy_blob_meta(result, temp);
   sys_sleep(1000);
   enqueue(ctxt->result_queue, result); 
   free_blob(result);
}

void steal_and_process_thread(void *arg){
   /*Check gateway for possible stealing victims*/
   device_ctxt* ctxt = (device_ctxt*)arg;
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
      close_service_connection(conn);
      if(temp->id == -1){
         free_blob(temp);
         sys_sleep(100);
         continue;
      }
      process_task(temp, ctxt);
      free_blob(temp);
   }
}

void generate_and_process_thread(void *arg){
   device_ctxt* ctxt = (device_ctxt*)arg;
   uint32_t task;
   blob* temp;
   char data[20] = "input_data";
   uint32_t frame_num;
   for(frame_num = 0; frame_num < ctxt->total_frames; frame_num ++){
      register_client(ctxt);
      for(task = 0; task < ctxt->batch_size; task ++){
         temp = new_blob_and_copy_data((int32_t)task, 20, (uint8_t*)data);
         annotate_blob(temp, get_this_client_id(ctxt), frame_num, task);
         enqueue(ctxt->task_queue, temp);
         free_blob(temp);
      }
      while(1){
         temp = try_dequeue(ctxt->task_queue);
         if(temp == NULL) break;
         process_task(temp, ctxt);
         free_blob(temp);
      }
      cancel_client(ctxt);
   }

}

void send_result_thread(void *arg){
   service_conn* conn;
   device_ctxt* ctxt = (device_ctxt*)arg;
   blob* temp;
#if DEBUG_FLAG
   uint32_t task_counter = 0;   
#endif
   while(1){
      temp = dequeue(ctxt->result_queue);
      conn = connect_service(TCP, ctxt->gateway_local_addr, RESULT_COLLECT_PORT);
      send_request("result_gateway", 20, conn);
#if DEBUG_FLAG
      task_counter ++;  
      printf("send_result for task %d:%d, total number is %d\n", get_blob_cli_id(temp), get_blob_task_id(temp), task_counter); 
#endif
      send_data(temp, conn);
      free_blob(temp);
      close_service_connection(conn);
   }
}

void* steal_client(void* srv_conn, void* arg){
   printf("steal_client ... ... \n");
   service_conn *conn = (service_conn *)srv_conn;
   device_ctxt* ctxt = (device_ctxt*)arg;
   blob* temp = try_dequeue(ctxt->task_queue);
   if(temp == NULL){
      char data[20]="empty";
      temp = new_blob_and_copy_data(-1, 20, (uint8_t*)data);
   }
#if DEBUG_FLAG
   printf("Stolen local task is %d\n", temp->id);
#endif
   send_data(temp, conn);
   free_blob(temp);
   return NULL;
}

void serve_stealing_thread(void *arg){
   const char* request_types[]={"steal_client"};
   void* (*handlers[])(void*, void*) = {steal_client};
   int wst_service = service_init(WORK_STEAL_PORT, TCP);
   start_service(wst_service, TCP, request_types, 1, handlers, arg);
   close_service(wst_service);
}



