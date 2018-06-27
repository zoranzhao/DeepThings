#include "darkiot.h"
#include "frame_partitioner.h"

void partition_and_enqueue(cnn_model* model, uint32_t frame_num){
   uint32_t task;
   network net = *(model->net);
   float* data;
   uint32_t data_size;
   blob* temp;
   uint32_t dw1, dw2;
   uint32_t dh1, dh2;
   uint32_t i, j;
   for(i = 0; i < model->ftp_para->partitions_h; i++){
      for(j = 0; j < model->ftp_para->partitions_w; j++){
         task = model->ftp_para->task_id[i][j];
         dw1 = model->ftp_para->input_tiles[task][0].w1;
         dw2 = model->ftp_para->input_tiles[task][0].w2;
         dh1 = model->ftp_para->input_tiles[task][0].h1;
         dh2 = model->ftp_para->input_tiles[task][0].h2;
         data = crop_feature_maps(net.input, net.layers[0].w, net.layers[0].h, net.layers[0].c, 
                                  dw1, dw2, dh1, dh2);
         data_size = sizeof(float)*(dw2-dw1+1)*(dh2-dh1+1)*net.layers[0].c;
         temp = new_blob_and_copy_data((int32_t)task, data_size, (uint8_t*)data);
         annotate_blob(temp, get_this_client_id(), frame_num, task);
         enqueue(task_queue, temp);
      }

   }
}

float* dequeue_and_merge(cnn_model* model){
   /*Check if there is a data frame whose tasks have all been collected*/
   blob* temp = dequeue(ready_pool);
   int32_t cli_id = temp->id;
   free_blob(temp);
#if DEBUG_FLAG
   printf("Results for client %d are all collected\n", cli_id);
#endif
   network net = *(model->net);
   ftp_parameters *ftp_para = model->ftp_para;
   uint32_t stage_outs =  (net.layers[ftp_para->fused_layers-1].out_w)*(net.layers[ftp_para->fused_layers-1].out_h)*(net.layers[ftp_para->fused_layers-1].out_c);
   float* stage_out = (float*) malloc(sizeof(float)*stage_outs);  
   uint32_t part = 0;
   uint32_t task = 0;
   for(part = 0; part < ftp_para->partitions; part ++){
      temp = dequeue(results_pool[cli_id]);
      task = get_blob_task_id(temp);
      stitch_feature_maps((float*)temp->data, stage_out, 
                          net.layers[ftp_para->fused_layers-1].out_w, 
                          net.layers[ftp_para->fused_layers-1].out_h, 
                          net.layers[ftp_para->fused_layers-1].out_c, 
                          ftp_para->output_tiles[task][ftp_para->fused_layers-1].w1, 
                          ftp_para->output_tiles[task][ftp_para->fused_layers-1].w2,
                          ftp_para->output_tiles[task][ftp_para->fused_layers-1].h1, 
                          ftp_para->output_tiles[task][ftp_para->fused_layers-1].h2);
      free_blob(temp);
   }
   return stage_out;
}

