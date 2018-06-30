#include "reuse_data_serialization.h"
bool* check_local_coverage(cnn_model* model, uint32_t task_id, uint32_t frame_num){
   ftp_parameters_reuse* ftp_para_reuse = model->ftp_para_reuse;

   uint32_t j = task_id%ftp_para_reuse->partitions_w;
   uint32_t i = task_id/ftp_para_reuse->partitions_w;
   uint32_t pos;
   bool* reuse_data_is_required = (bool*) malloc(4*sizeof(bool));
   /*position encoding
         2
         |
   3 <- self -> 1
         |
         0
   */
   for(pos = 0; pos < 4; pos++){
      reuse_data_is_required[pos] = true;
   }
   /*check the up overlapped data from tile below*/
   if((i+1)<(ftp_para_reuse->partitions_h)){
      if(get_coverage(ftp_para_reuse, ftp_para_reuse->task_id[i+1][j])==1) reuse_data_is_required[0] = false;
   }else{reuse_data_is_required[0] = false;}
   /*check the left overlapped data from tile on the right*/
   if((j+1)<(ftp_para_reuse->partitions_w)) {
      if(get_coverage(ftp_para_reuse, ftp_para_reuse->task_id[i][j+1])==1) reuse_data_is_required[1] = false;
   }else{reuse_data_is_required[1] = false;}
   /*check the bottom overlapped data from tile above*/
   if(i>0){
      if(get_coverage(ftp_para_reuse, ftp_para_reuse->task_id[i-1][j])==1) reuse_data_is_required[2] = false;
   }else{reuse_data_is_required[2] = false;}
   /*check the right overlapped data from tile on the left*/
   if(j>0){
      if(get_coverage(ftp_para_reuse, ftp_para_reuse->task_id[i][j-1])==1) reuse_data_is_required[3] = false;
   }else{reuse_data_is_required[3] = false;}
   return reuse_data_is_required;

}

blob* reuse_data_serialization(cnn_model* model, uint32_t task_id, uint32_t frame_num, bool *reuse_data_is_required){
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

   for(position = 0; position < 4; position++) 
      adjacent_id[position]=-1;

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
         if(reuse_data_is_required[position]==false) continue;
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

overlapped_tile_data** reuse_data_deserialization(cnn_model* model, uint32_t task_id, float* input, uint32_t frame_num, bool *reuse_data_is_required){
   ftp_parameters_reuse* ftp_para_reuse = model->ftp_para_reuse;
   network_parameters* net_para = model->net_para;

   tile_region overlap_index;
   uint32_t i = task_id / (ftp_para_reuse->partitions_w); 
   uint32_t j = task_id % (ftp_para_reuse->partitions_w);
   int32_t adjacent_id[4];
   uint32_t position;
   uint32_t l;
   float* serial_data = input;

   overlapped_tile_data** regions_and_data_ptr_array = (overlapped_tile_data**)malloc(sizeof(overlapped_tile_data*)*(4));
   for(position = 0; position < 4; position++){
      adjacent_id[position]=-1;
      regions_and_data_ptr_array[position] = (overlapped_tile_data*)malloc(sizeof(overlapped_tile_data)*(ftp_para_reuse->fused_layers));
   }

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
         if(reuse_data_is_required[position]==false) continue;
         uint32_t mirror_position = (position + 2)%4;
         regions_and_data_ptr_array[position][l] = ftp_para_reuse->output_reuse_regions[adjacent_id[position]][l];
         overlapped_tile_data* regions_and_data_ptr = regions_and_data_ptr_array[position];
         overlap_index = get_region(regions_and_data_ptr+l, mirror_position);
         if((overlap_index.w>0)&&(overlap_index.h>0)){
            if(get_size(regions_and_data_ptr+l, mirror_position)>0) free(get_data(regions_and_data_ptr+l, mirror_position));
            uint32_t amount_of_element = overlap_index.w*overlap_index.h*net_para->output_maps[l].c;
            float* data = (float* )malloc(amount_of_element*sizeof(float));
            memcpy(data, serial_data, amount_of_element*sizeof(float)); 
            serial_data = serial_data + amount_of_element;
            set_size(regions_and_data_ptr+l, mirror_position, amount_of_element*sizeof(float));
            set_data(regions_and_data_ptr+l, mirror_position, data);
         }
      }
   }

   return regions_and_data_ptr_array;
}

void free_overlapped_tile_data_ptr_array(overlapped_tile_data** regions_and_data_ptr_array){

   uint32_t position;
   for(position = 0; position < 4; position++){
      free(regions_and_data_ptr_array[position]);
   }
   free(regions_and_data_ptr_array);
}


