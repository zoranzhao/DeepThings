#ifndef DEEPTHINGS_ANNOTATION_H
#define DEEPTHINGS_ANNOTATION_H
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <sys/time.h>
#include <stdbool.h> 

/*Include necessary context objects from deepthings and darkiot*/
#include "deepthings_edge.h"
#include "ftp.h"
#include "inference_engine_helper.h"
/*Include necessary context objects from deepthings and darkiot*/

#ifndef PARTITIONS_W_MAX
#define PARTITIONS_W_MAX 6
#endif

#ifndef PARTITIONS_H_MAX
#define PARTITIONS_H_MAX 6
#endif

#ifndef PARTITIONS_MAX
#define PARTITIONS_MAX 36
#endif

#ifndef FRAME_NUM
#define FRAME_NUM 4
#endif

#ifndef FUSED_LAYERS_MAX
#define FUSED_LAYERS_MAX 16
#endif

#ifndef NUM_OF_FUNCTIONS
#define NUM_OF_FUNCTIONS 9
#endif

typedef struct def_deepthings_annotation_data {
  bool valid[FRAME_NUM][PARTITIONS_MAX][2];
  double avg_duration[FRAME_NUM][PARTITIONS_MAX][2]; /*0 no data-reuse, 1 data-reuse*/
} deepthings_annotation_data;

void simulation_start(device_ctxt* ctxt, void* sim_ctxt);
void simulation_end(device_ctxt* ctxt, void* sim_ctxt);
void function_delay(char* function_name, device_ctxt* ctxt, void* sim_ctxt, uint32_t frame_number, uint32_t partition_number, uint32_t data_reuse);

#endif/*DEEPTHINGS_ANNOTATION_H*/
