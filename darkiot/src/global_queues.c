#include "global_queues.h"
thread_safe_queue** results_pool;
thread_safe_queue* ready_pool;
uint32_t* results_counter;
thread_safe_queue* registration_list;
thread_safe_queue* task_queue;
thread_safe_queue* result_queue;

void init_queues(uint32_t cli_num){
   results_pool = (thread_safe_queue**)malloc(sizeof(thread_safe_queue*)*cli_num);
   results_counter = (uint32_t*)malloc(sizeof(uint32_t)*cli_num);
   uint32_t i;
   for(i = 0; i < cli_num; i++){
      results_pool[i] = new_queue(MAX_QUEUE_SIZE);
      results_counter[i] = 0;
   } 
   ready_pool = new_queue(MAX_QUEUE_SIZE); 
   registration_list = new_queue(MAX_QUEUE_SIZE); 

   task_queue = new_queue(MAX_QUEUE_SIZE);
   result_queue = new_queue(MAX_QUEUE_SIZE); 
}
