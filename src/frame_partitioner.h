#ifndef FRAME_PARTITIONER_H 
#define FRAME_PARTITIONER_H
#include "ftp.h"
#include "inference_engine_helper.h"
void partition_and_enqueue(device_ctxt* ctxt, uint32_t frame_num);
blob* dequeue_and_merge(device_ctxt* ctxt);
#endif
