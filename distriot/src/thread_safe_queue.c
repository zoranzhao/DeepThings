#include "thread_safe_queue.h"

static queue_node* new_node_and_copy_item(blob* item)
{
   queue_node *temp = (queue_node*)malloc(sizeof(queue_node));
   temp->item = new_blob_and_copy_data(item->id, item->size, item->data);
   copy_blob_meta(temp->item, item);
   temp->next = NULL;
   return temp; 
}

thread_safe_queue *new_queue(uint32_t capacity)
{
   thread_safe_queue *q = (thread_safe_queue*)malloc(sizeof(thread_safe_queue));
   q->head = q->tail = NULL;
   q->capacity = capacity;
   q->number_of_node = 0;
   sys_sem_new(&(q->not_empty), 0);
   sys_sem_new(&(q->not_full), 0);
   sys_sem_new(&(q->mutex), 1);
   q->wait_send = 0;
   return q;
}

void enqueue(thread_safe_queue *queue, blob* item)
{
   uint8_t first;
   queue_node *temp = new_node_and_copy_item(item);
   sys_arch_sem_wait(&(queue->mutex), 0);
   while ((queue->number_of_node) >= (queue->capacity)) {
      queue->wait_send++;
      sys_sem_signal(&queue->mutex);
      sys_arch_sem_wait(&queue->not_full, 0);
      sys_arch_sem_wait(&queue->mutex, 0);
      queue->wait_send--;
   }

   if (queue->tail == NULL){
      queue->head = queue->tail = temp;
      first = 1;
   }else{
      queue->tail->next = temp;
      queue->tail = temp;
      first = 0;
   }
   queue->number_of_node++;

   if (first) {
      sys_sem_signal(&queue->not_empty);
   }

   sys_sem_signal(&queue->mutex);

}

blob* dequeue(thread_safe_queue *queue)
{
   sys_arch_sem_wait(&queue->mutex, 0);
   while (queue->head == NULL) {
      sys_sem_signal(&queue->mutex);
      /* We block while waiting for a mail to arrive in the mailbox. */
      sys_arch_sem_wait(&queue->not_empty, 0);
      sys_arch_sem_wait(&queue->mutex, 0);
   } 

   queue_node *temp = queue->head;
   queue->head = queue->head->next;
   if (queue->head == NULL)
      queue->tail = NULL;
   blob* item = temp->item; 
   free(temp);
   queue->number_of_node--;

   if (queue->wait_send) {
      sys_sem_signal(&queue->not_full);
   }

   sys_sem_signal(&queue->mutex);

   return item;
}

void print_queue_by_id(thread_safe_queue *queue){
   sys_arch_sem_wait(&queue->mutex, 0);
   queue_node* cur = queue->head;
   if (queue->head == NULL){
      sys_sem_signal(&queue->mutex);
      return;
   }
   while (1) {
      if (cur->next == NULL){
         printf("%d\n", cur->item->id);
         break;
      }
      printf("%d, ", cur->item->id);
      cur = cur->next;
   } 

   sys_sem_signal(&queue->mutex);
   return;
}

void remove_by_id(thread_safe_queue *queue, int32_t id){
   sys_arch_sem_wait(&queue->mutex, 0);
   queue_node* prev = queue->head;
   queue_node* cur = queue->head->next;
   if (queue->head == NULL){
      sys_sem_signal(&queue->mutex);
      return;
   }

   if(queue->head->item->id == id){
      queue->head = queue->head->next;
      free_blob(prev->item);
      free(prev);
      queue->number_of_node--;
      if (queue->head == NULL)
         queue->tail = NULL;
      sys_sem_signal(&queue->mutex);
      return;
   }
   
   while (cur != NULL) {
      if(cur->item->id == id){
         prev->next = cur->next;
         if(cur->next == NULL){
            queue->tail=prev;
         }
         free_blob(cur->item);
         free(cur);
         queue->number_of_node--;
         sys_sem_signal(&queue->mutex);
	 return;
      }
      prev = cur;
      cur = cur->next;
   } 

   sys_sem_signal(&queue->mutex);
   return;
}

blob* try_dequeue(thread_safe_queue *queue)
{
   sys_arch_sem_wait(&queue->mutex, 0);
   while (queue->head == NULL) {
      sys_sem_signal(&queue->mutex);
      return NULL;
   } 

   queue_node *temp = queue->head;
   queue->head = queue->head->next;
   if (queue->head == NULL)
      queue->tail = NULL;
   blob* item = temp->item; 
   free(temp);
   queue->number_of_node--;

   if (queue->wait_send) {
      sys_sem_signal(&queue->not_full);
   }

   sys_sem_signal(&queue->mutex);

   return item;
}

void free_queue(thread_safe_queue *queue)
{
  if (queue != NULL) {
    sys_arch_sem_wait(&queue->mutex, 0);
    sys_sem_free(&queue->not_empty);
    sys_sem_free(&queue->not_full);
    sys_sem_free(&queue->mutex);
    queue->not_empty = queue->not_full = queue->mutex = NULL;
    free(queue);
  }
}


