#ifndef DEEPTHINGS_EDGE_H
#define DEEPTHINGS_EDGE_H
#include "darkiot.h"
#include "configure.h"

void deepthings_stealer_edge(uint32_t N, uint32_t M, uint32_t fused_layers, char* network, char* weights);
void deepthings_victim_edge(uint32_t N, uint32_t M, uint32_t fused_layers, char* network, char* weights);

#endif
