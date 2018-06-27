#ifndef CONFIGURE_H
#define CONFIGURE_H

/*Partitioning paramters*/
#define FUSED_LAYERS_MAX 16
#define PARTITIONS_W_MAX 6
#define PARTITIONS_H_MAX 6
#define PARTITIONS_MAX 36
#define THREAD_NUM 1
#include <stdint.h>

extern uint32_t fused_layers;
extern uint32_t partitions;
extern uint32_t partitions_w;
extern uint32_t partitions_h;

#endif
