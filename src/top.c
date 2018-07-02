#include "darkiot.h"
#include "config.h"
#include "configure.h"
#include "ftp.h"
#include "inference_engine_helper.h"
#include "frame_partitioner.h"
#include "reuse_data_serialization.h"
#include <string.h>
#include "deepthings_edge.h"
#include "deepthings_gateway.h"

/*
make ARGS="2 start" test
make ARGS="2 wst gateway" test
make ARGS="0 wst_s data_source" test
make ARGS="1 wst non_data_source" test
make ARGS="2 wst non_data_source" test
*/


void test_wst(int argc, char **argv)
{
   printf("total_cli_num %d\n", atoi(argv[1]));
   printf("this_cli_id %d\n", atoi(argv[1]));

   this_cli_id = atoi(argv[1]);
   total_cli_num = atoi(argv[1]);
      
   if(0 == strcmp(argv[2], "start")){  
      printf("start\n");
      exec_start_gateway(START_CTRL, TCP);
   }else if(0 == strcmp(argv[2], "wst")){
      printf("Work stealing\n");
      if(0 == strcmp(argv[3], "non_data_source")){
         printf("non_data_source\n");
         deepthings_stealer_edge();
      }else if(0 == strcmp(argv[3], "data_source")){
         printf("data_source\n");
         deepthings_victim_edge();
      }else if(0 == strcmp(argv[3], "gateway")){
         printf("gateway\n");
         deepthings_gateway();
      }
   }else if(0 == strcmp(argv[2], "wst_s")){
      printf("Work stealing with scheduling\n");
      if(0 == strcmp(argv[3], "non_data_source")){
         printf("non_data_source\n");
         deepthings_stealer_edge();
      }else if(0 == strcmp(argv[3], "data_source")){
         printf("data_source\n");
         deepthings_victim_edge();
      }else if(0 == strcmp(argv[3], "gateway")){
         printf("gateway\n");
         deepthings_gateway();
      }
   }
}
void local_ftp(int argc, char **argv){
   this_cli_id = 0;
   total_cli_num = 1;
   init_queues(total_cli_num);
#if DATA_REUSE
   shrinked_task_queue = new_queue(MAX_QUEUE_SIZE);
   schedule_task_queue = new_queue(MAX_QUEUE_SIZE);
#endif

   cnn_model* model = load_cnn_model((char*)"models/yolo.cfg", (char*)"models/yolo.weights");
   model->ftp_para = preform_ftp(3, 3, 4, model->net_para);
#if DATA_REUSE
   model->ftp_para_reuse = preform_ftp_reuse(model->net_para, model->ftp_para);
#endif
/*
   for(int i = 0; i < 2; i++){
      for(int j = 0; j < 2; j++){
         printf("------------------(%3d,%3d)----------------\n", i, j);
         for(int l = 2; l >= 0; l--)
            print_tile_region(model->ftp_para_reuse->output_tiles[model->ftp_para->task_id[i][j]][l]);
         print_tile_region(model->ftp_para_reuse->input_tiles[model->ftp_para->task_id[i][j]][0]);
      }
   }
*/

   for(int frame_seq = 0; frame_seq < 4; frame_seq++){
      image_holder img = load_image_as_model_input(model, frame_seq);
      partition_and_enqueue(model, frame_seq);
      blob* temp;
      blob* result;
      while(1){
         temp = try_dequeue(task_queue);
         if(temp == NULL) break;
         printf("Task id is %d\n", temp->id);
#if DATA_REUSE
         if(model->ftp_para_reuse->schedule[get_blob_task_id(temp)] == 1 && is_reuse_ready(model->ftp_para_reuse, get_blob_task_id(temp))) {
             set_model_input(model, (float*)(model->ftp_para_reuse->shrinked_input[get_blob_task_id(temp)]));
             printf("Reuse ... %d\n", temp->id);
             //clean_coverage(model->ftp_para_reuse);
             model->ftp_para_reuse->coverage[0] = 0;
             model->ftp_para_reuse->coverage[2] = 0;
             model->ftp_para_reuse->coverage[4] = 0;
             model->ftp_para_reuse->coverage[6] = 0;
             model->ftp_para_reuse->coverage[8] = 0;
             model->ftp_para_reuse->coverage[10] = 0;
             model->ftp_para_reuse->coverage[12] = 0;
             model->ftp_para_reuse->coverage[14] = 0;
             model->ftp_para_reuse->coverage[16] = 0;
             model->ftp_para_reuse->coverage[18] = 0;
             model->ftp_para_reuse->coverage[20] = 0;
             model->ftp_para_reuse->coverage[22] = 0;
             model->ftp_para_reuse->coverage[24] = 0;
             bool* tmp = check_local_coverage(model, temp->id, frame_seq);
             printf("Down %d\n",  tmp[0]);
             printf("Right %d\n", tmp[1]);
             printf("Up %d\n",    tmp[2]);
             printf("Left %d\n",  tmp[3]);
             blob* reuse_blob = adjacent_reuse_data_serialization(model, temp->id, frame_seq, tmp); 
             overlapped_tile_data** temp_region_and_data = adjacent_reuse_data_deserialization(model, temp->id, (float*)reuse_blob->data, frame_seq, tmp);
             place_adjacent_deserialized_data(model, temp->id, temp_region_and_data, tmp);
             free(tmp);
             free_overlapped_tile_data_ptr_array(temp_region_and_data);
             free_blob(reuse_blob);
             set_coverage(model->ftp_para_reuse, 0);
             set_coverage(model->ftp_para_reuse, 2);
             set_coverage(model->ftp_para_reuse, 4);
             set_coverage(model->ftp_para_reuse, 6);
             set_coverage(model->ftp_para_reuse, 8);
             set_coverage(model->ftp_para_reuse, 10);
             set_coverage(model->ftp_para_reuse, 12);
             set_coverage(model->ftp_para_reuse, 14);
             set_coverage(model->ftp_para_reuse, 16);
             set_coverage(model->ftp_para_reuse, 18);
             set_coverage(model->ftp_para_reuse, 20);
             set_coverage(model->ftp_para_reuse, 22);
             set_coverage(model->ftp_para_reuse, 24);
         }
         else 
#endif
         set_model_input(model, (float*)temp->data);

         forward_partition(model, get_blob_task_id(temp));  
         result = new_blob_and_copy_data(0, 
                                      get_model_byte_size(model, model->ftp_para->fused_layers-1), 
                                      (uint8_t*)(get_model_output(model, model->ftp_para->fused_layers-1))
                                     );
         copy_blob_meta(result, temp);
         enqueue(results_pool[this_cli_id], result);
         //enqueue(result_queue, result); 
#if DATA_REUSE
         if(model->ftp_para_reuse->schedule[get_blob_task_id(temp)] == 0) {
               blob* self_ir  = self_reuse_data_serialization(model, get_blob_task_id(temp), frame_seq);
               overlapped_tile_data* temp_region_and_data = self_reuse_data_deserialization(model, get_blob_task_id(temp), (float*)self_ir->data, frame_seq);
               place_self_deserialized_data(model, get_blob_task_id(temp), temp_region_and_data);
         }
#endif
         free_blob(result);
         free_blob(temp);
      }

      enqueue(ready_pool, new_empty_blob(this_cli_id));
      temp = (dequeue_and_merge(model));
      float* fused_output = (float* )temp->data;
      set_model_input(model, fused_output);
      forward_all(model, model->ftp_para->fused_layers);   
      free(fused_output);
      draw_object_boxes(model, frame_seq);
      free_image_holder(model, img);
   }
}


int main(int argc, char **argv){
   /*local_ftp(argc, argv);*/
   test_wst(argc, argv);
   return 0;
}

