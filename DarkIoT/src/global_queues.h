#ifndef GLOBAL_QUEUES_H
#define GLOBAL_QUEUES_H
#include "thread_safe_queue.h"
#include "config.h"

extern thread_safe_queue** results_pool;
extern thread_safe_queue* ready_pool;
extern uint32_t* results_counter;
extern thread_safe_queue* registration_list;
extern thread_safe_queue* task_queue;
extern thread_safe_queue* result_queue;

void init_queues(uint32_t cli_num);

#endif

