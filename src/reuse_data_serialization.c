#include "reuse_data_serialization.h"

blob* reuse_data_serialization(cnn_model* model, uint32_t task_id, uint32_t frame_num){
   ftp_parameters_reuse* ftp_para_reuse = model->ftp_para_reuse;
   network_parameters* net_para = model->net_para;
   overlapped_tile_data regions_and_data;
   tile_region overlap_index;
   uint32_t i = task_id / (ftp_para_reuse->partitions_w); 
   uint32_t j = task_id % (ftp_para_reuse->partitions_w);
   int32_t adjacent_id[4];
   uint32_t position;
   float *reuse_data;
   uint32_t size = 0;
   uint32_t l;
   reuse_data = (float*)malloc(ftp_para_reuse->reuse_data_size[task_id]);
   /*get the up overlapped data from tile below*/
   if((i+1)<(ftp_para_reuse->partitions_h)) adjacent_id[0] = ftp_para_reuse->task_id[i+1][j];
   /*get the left overlapped data from tile on the right*/
   if((j+1)<(ftp_para_reuse->partitions_w)) adjacent_id[1] = ftp_para_reuse->task_id[i][j+1];
   /*get the bottom overlapped data from tile above*/
   if(i>0) adjacent_id[2] = ftp_para_reuse->task_id[i-1][j];
   /*get the right overlapped data from tile on the left*/
   if(j>0) adjacent_id[3] = ftp_para_reuse->task_id[i][j-1];

   for(l = 0; l < ftp_para_reuse->fused_layers-1; l ++){
      for(position = 0; position < 4; position++){
         if(adjacent_id[position]==-1) continue;
         uint32_t mirror_position = (position + 2)%4;
         regions_and_data = ftp_para_reuse->output_reuse_regions[adjacent_id[position]][l];
         overlap_index = get_region(&regions_and_data, mirror_position);
         if((overlap_index.w>0)&&(overlap_index.h>0)){
            uint32_t amount_of_element = overlap_index.w*overlap_index.h*net_para->output_maps[l].c;
            memcpy(reuse_data, get_data(&regions_and_data, mirror_position), amount_of_element*sizeof(float) ); 
            reuse_data = reuse_data + amount_of_element;
            size += amount_of_element;

         }
      }
   }
   reuse_data = reuse_data - size;
   size = (size) * sizeof(float);
   blob* temp = new_blob_and_copy_data((int32_t)task_id, size, (uint8_t*)reuse_data);
   annotate_blob(temp, get_this_client_id(), frame_num, task_id);
   free(reuse_data);
   return temp;
}


