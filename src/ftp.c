#include "ftp.h"
#include "inference_engine_helper.h"

static inline void grid(network_parameters* net_para, ftp_parameters* ftp_para, uint32_t M, uint32_t N){
   int32_t w = net_para->output_maps[ftp_para->fused_layers-1].w;
   int32_t h = net_para->output_maps[ftp_para->fused_layers-1].h;
   int32_t partition_w = M;
   int32_t partition_h = N;
   int32_t stride_w = ceil(((float)w)/((float)partition_w));    
   int32_t start_w = 0;
   int32_t end_w = stride_w - 1;
   int32_t stride_h = ceil(((float)h)/((float)partition_h));    
   int32_t start_h = 0;
   int32_t end_h = stride_h - 1;
   int32_t i, j, task_id;

   for(i = 0; i < partition_h; i++){
      start_w = 0;
      end_w = stride_w - 1;	 
      for(j = 0; j < partition_w; j++){
         task_id = ftp_para->task_id[i][j];
         ftp_para->output_tiles[task_id][ftp_para->fused_layers-1].w1 = start_w;
         ftp_para->output_tiles[task_id][ftp_para->fused_layers-1].w2 = end_w;
         ftp_para->output_tiles[task_id][ftp_para->fused_layers-1].h1 = start_h;
         ftp_para->output_tiles[task_id][ftp_para->fused_layers-1].h2 = end_h;
         ftp_para->output_tiles[task_id][ftp_para->fused_layers-1].h = end_h - start_h + 1;
         ftp_para->output_tiles[task_id][ftp_para->fused_layers-1].w = end_w - start_w + 1;
         /*print_tile_region(ftp_para->output_tiles[task_id][ftp_para->fused_layers-1]);*/
         start_w = end_w + 1;
         if(j == (partition_w-2)) {end_w = w - 1;}
         else {end_w = end_w + stride_w;}
      }
      start_h = end_h + 1;
      if(i == (partition_h-2)) {end_h = h - 1;}
      else {end_h = end_h + stride_h;}
   }

}

static inline void traversal(network_parameters* net_para, ftp_parameters* ftp_para, uint32_t l, uint32_t i, uint32_t j){
   tile_region input; 
   tile_region output = ftp_para->output_tiles[ftp_para->task_id[i][j]][l]; 
   int32_t stride = net_para->stride[l];
   int32_t filter = net_para->filter[l];    
   int32_t w = net_para->input_maps[l].w;
   int32_t h = net_para->input_maps[l].h;     

   if(net_para->type[l] == CONV_LAYER){
      input.w1 = (output.w1*stride - filter/2)>0 ? (output.w1*stride - filter/2) : 0;
      input.w2 = (output.w2*stride + filter/2)<(w-1) ? (output.w2*stride + filter/2) : (w-1);
      input.h1 = (output.h1*stride - filter/2)>0   ? (output.h1*stride - filter/2) : 0;
      input.h2 = (output.h2*stride + filter/2)<(h-1) ? (output.h2*stride + filter/2) : (h-1);
   }else if(net_para->type[l] == POOLING_LAYER){
      input.w1 = output.w1*stride;
      input.w2 = output.w2*stride + stride -1;
      input.h1 = output.h1*stride;
      input.h2 = output.h2*stride + stride -1;
   }
   input.w = input.w2 -input.w1 + 1;
   input.h = input.h2 -input.h1 + 1;
   ftp_para->input_tiles[ftp_para->task_id[i][j]][l] = input; 
}

ftp_parameters* preform_ftp(uint32_t N, uint32_t M, uint32_t fused_layers, network_parameters* net_para){
   ftp_parameters* ftp_para = (ftp_parameters*)malloc(sizeof(ftp_parameters));
   ftp_para->partitions = N*M;
   ftp_para->partitions_h = N;
   ftp_para->partitions_w = M;
   ftp_para->fused_layers = fused_layers;
   int32_t i, j, l;
   int32_t id = 0;
   for(i = 0; i < ftp_para->partitions_h; i++){
      for(j = 0; j < ftp_para->partitions_w; j++){
         ftp_para->task_id[i][j] = id;
         id++;
      }
   }
   grid(net_para, ftp_para, M, N);
   for(i = 0; i < ftp_para->partitions_h; i++){
      for(j = 0; j < ftp_para->partitions_w; j++){
         for(l = fused_layers-1; l >= 0; l--){
            traversal(net_para, ftp_para, l, i, j);
            if(l>0) ftp_para->output_tiles[ftp_para->task_id[i][j]][l-1] 
                     = ftp_para->input_tiles[ftp_para->task_id[i][j]][l];
         }
      }
   }
   return ftp_para;
}

void print_tile_region(tile_region tile){
   printf("tile size is (%3d,%3d) \n", tile.w, tile.h);
   printf("(%3d,%3d)--------|\n", tile.w1, tile.h1);
   printf("|----------------|\n");
   printf("|----------------|\n");
   printf("|--------(%3d,%3d)\n", tile.w2, tile.h2);
}

