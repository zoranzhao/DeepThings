#ifndef FRAME_PARTITIONER_H 
#define FRAME_PARTITIONER_H
#include "ftp.h"
#include "inference_engine_helper.h"
void partition_and_enqueue(cnn_model* model, uint32_t frame_num);
blob* dequeue_and_merge(cnn_model* model);
#if DATA_REUSE
extern thread_safe_queue* shrinked_task_queue;
extern thread_safe_queue* schedule_task_queue;
#endif
#endif
