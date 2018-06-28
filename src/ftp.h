#ifndef FTP_H
#define FTP_H
#include "configure.h"

typedef struct partition_range{
    int32_t w1;
    int32_t h1;
    int32_t w2;
    int32_t h2;
    int32_t h;
    int32_t w;
    int32_t c;/*Channel number*/
} tile_region;

typedef struct def_ftp_para{
   uint32_t partitions;
   uint32_t partitions_w;
   uint32_t partitions_h;
   uint32_t fused_layers;
   uint32_t task_id[PARTITIONS_H_MAX][PARTITIONS_W_MAX];
   tile_region input_tiles[PARTITIONS_MAX][FUSED_LAYERS_MAX];
   tile_region output_tiles[PARTITIONS_MAX][FUSED_LAYERS_MAX];
} ftp_parameters;

typedef struct def_network_para{
   uint32_t layers;
   uint32_t *stride;
   uint32_t *filter;
   uint32_t *type;
   tile_region *input_maps;
   tile_region *output_maps;
} network_parameters;

#if DATA_REUSE
typedef struct def_overlapped_data{
   float *down;
   float *right;
   float *up;
   float *left;
   tile_region down_region;
   tile_region right_region;
   tile_region left_region;
   tile_region up_region;
} overlapped_tile_data;

typedef struct def_ftp_parameters_reuse{
   uint32_t partitions;
   uint32_t partitions_w;
   uint32_t partitions_h;
   uint32_t fused_layers;
   uint32_t task_id[PARTITIONS_H_MAX][PARTITIONS_W_MAX];
   uint32_t schedule[PARTITIONS_H_MAX][PARTITIONS_W_MAX];
   tile_region input_tiles[PARTITIONS_MAX][FUSED_LAYERS_MAX];
   tile_region output_tiles[PARTITIONS_MAX][FUSED_LAYERS_MAX];
   overlapped_tile_data output_reuse_regions[PARTITIONS_MAX][FUSED_LAYERS_MAX];
} ftp_parameters_reuse;
#endif

ftp_parameters* preform_ftp(uint32_t N, uint32_t M, uint32_t fused_layers, network_parameters* net_para);
void print_tile_region(tile_region tile);
#endif
