#ifndef DEEPTHINGS_EDGE_H
#define DEEPTHINGS_EDGE_H
#include "darkiot.h"
#include "configure.h"
device_ctxt* deepthings_edge_init(uint32_t N, uint32_t M, uint32_t fused_layers, char* network, char* weights, uint32_t edge_id);
void deepthings_stealer_edge(uint32_t N, uint32_t M, uint32_t fused_layers, char* network, char* weights, uint32_t edge_id);
void deepthings_victim_edge(uint32_t N, uint32_t M, uint32_t fused_layers, char* network, char* weights, uint32_t edge_id);
void partition_frame_and_perform_inference_thread(void *arg);
void steal_partition_and_perform_inference_thread(void *arg);
void deepthings_serve_stealing_thread(void *arg);
#endif
