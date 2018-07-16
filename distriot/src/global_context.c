#include "global_context.h"
/*Queues used in gateway device*/
/*
thread_safe_queue** results_pool;
thread_safe_queue* ready_pool;
uint32_t* results_counter;
thread_safe_queue* registration_list;
*/

/*Queues used in edge device*/
/*
thread_safe_queue* task_queue;
thread_safe_queue* result_queue;
static const char* edge_addr_list[CLI_NUM] = EDGE_ADDR_LIST;
*/



device_ctxt* init_context(uint32_t cli_id, uint32_t cli_num, const char** edge_addr_list){

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

/*Queues used in edge device*/
   ctxt->task_queue = new_queue(MAX_QUEUE_SIZE);
   ctxt->result_queue = new_queue(MAX_QUEUE_SIZE); 
   ctxt->this_cli_id = cli_id;

   return ctxt;
}

void set_batch_size(device_ctxt* ctxt, uint32_t size){
   ctxt->batch_size = size;
}

void set_gateway_local_addr(device_ctxt* ctxt, const char* addr){
   strcpy(ctxt->gateway_local_addr, addr);
}

void set_gateway_public_addr(device_ctxt* ctxt, const char* addr){
   strcpy(ctxt->gateway_public_addr, addr);
}

void set_total_frames(device_ctxt* ctxt, uint32_t frame_num){
   ctxt->total_frames = frame_num;
}



