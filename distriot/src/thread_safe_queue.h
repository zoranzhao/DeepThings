#ifndef THREAD_SAFE_QUEUE_H
#define THREAD_SAFE_QUEUE_H
#include "thread_util.h"
#include "data_blob.h"
#include "stdio.h"

#ifdef __cplusplus
extern "C" {
#endif
typedef struct ts_node {
   blob* item;
   struct ts_node* next;
} queue_node;

/*Set a capacity to consider the limited memory space in IoT devices*/
typedef struct ts_queue {
   uint32_t capacity;
   uint32_t number_of_node;
   queue_node* head, *tail;
   sys_sem_t not_empty;
   sys_sem_t not_full;
   sys_sem_t mutex;
   uint32_t wait_send;
} thread_safe_queue;

thread_safe_queue *new_queue(uint32_t capacity);
void enqueue(thread_safe_queue *q, blob* item);
blob* dequeue(thread_safe_queue *q);
void remove_by_id(thread_safe_queue *q, int32_t id);
void print_queue_by_id(thread_safe_queue *queue);
blob* try_dequeue(thread_safe_queue *queue);
void free_queue(thread_safe_queue *q);
#ifdef __cplusplus
}
#endif

#endif
