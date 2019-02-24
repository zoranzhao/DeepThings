#include "deepthings_annotation.h"

static deepthings_annotation_data deepthings_annot_data[PARTITIONS_H_MAX][PARTITIONS_W_MAX][FUSED_LAYERS_MAX][NUM_OF_FUNCTIONS];

static double total_simulation_time;

static char function_list[NUM_OF_FUNCTIONS][40]={
/*Serialization functions used in edge node devices*/
   "self_reuse_data_serialization",
   "adjacent_reuse_data_deserialization",
   "place_adjacent_deserialized_data",

/*Serialization functions used in gateway devices*/
   "place_self_deserialized_data",
   "adjacent_reuse_data_serialization",
   "self_reuse_data_deserialization",

/*Data dependent function calls*/
   "forward_partition",
   "load_image_as_model_input",
   "partition_and_enqueue"
};

static inline uint32_t get_function_id(char* function_name){
   uint32_t id = 0; 
   for(id = 0; id < NUM_OF_FUNCTIONS; id++){
      if(strcmp(function_name, function_list[id]) == 0) return id;
   }
   return 0;
}

#define BUFFER_SIZE 200
void load_profile(uint32_t partitions_h, uint32_t partitions_w, uint32_t layers){
   char filename[50];
   sprintf(filename, "%dx%d_grid_%d_layers.prof", partitions_h, partitions_w, layers);
   const char *delimiter = "	";
   FILE *profile_data = fopen(filename, "r");
   char buffer[BUFFER_SIZE];
   char *token;
   uint32_t line_number = 0;
   uint32_t token_number = 0; 

   if(profile_data == NULL){
      printf("Unable to open file %s\n", filename);
   }else{
      while(fgets(buffer, BUFFER_SIZE, profile_data) != NULL){
         line_number++;
         if(line_number == 1) continue;
         token_number = 0;
         uint32_t function_id = 0;
         uint32_t frame_number = 0;
         uint32_t partition_number = 0;
         uint32_t data_reuse = 0;
         token = strtok(buffer, delimiter);
         while(token != NULL){
            token_number++;
	    switch (token_number){
               case 1: function_id = get_function_id(token); break;
               case 2: frame_number = atoi(token); break;
               case 3: partition_number = atoi(token); break;
               case 4: data_reuse = atoi(token); break;
               case 5: break; /*We can actually skip the calling times*/
               case 6: deepthings_annot_data[partitions_h][partitions_w][layers][function_id].avg_duration[frame_number][partition_number][data_reuse] = atof(token); 
                       deepthings_annot_data[partitions_h][partitions_w][layers][function_id].valid[frame_number][partition_number][data_reuse] = true;
                       break;
            }
            token = strtok(NULL, delimiter);
         }
      }
   }
}

void simulation_start(device_ctxt* ctxt, void* sim_ctxt){
   uint32_t function_id;
   uint32_t frame_number;
   uint32_t partition_number;

   cnn_model* model = (cnn_model*)ctxt->model;
   uint32_t partitions_h = model->ftp_para->partitions_h;
   uint32_t partitions_w = model->ftp_para->partitions_w;
   uint32_t layers = model->ftp_para->fused_layers;

   for(function_id = 0; function_id < NUM_OF_FUNCTIONS; function_id++){
      for(frame_number = 0; frame_number < FRAME_NUM; frame_number++){
         for(partition_number=0; partition_number<PARTITIONS_MAX; partition_number++){
            deepthings_annot_data[partitions_h][partitions_w][layers][function_id].valid[frame_number][partition_number][0] = false;
            deepthings_annot_data[partitions_h][partitions_w][layers][function_id].valid[frame_number][partition_number][1] = false;
            deepthings_annot_data[partitions_h][partitions_w][layers][function_id].avg_duration[frame_number][partition_number][0] = 0.0;
            deepthings_annot_data[partitions_h][partitions_w][layers][function_id].avg_duration[frame_number][partition_number][1] = 0.0;
         }
      }
   }
   /*Initialize the simulation context*/
   total_simulation_time = 0;
   /*Initialize the simulation context*/
}

void simulation_end(device_ctxt* ctxt, void* sim_ctxt){
   printf("Total simulation time is %f\n", total_simulation_time);
}

void function_delay(char* function_name, device_ctxt* ctxt, void* sim_ctxt, uint32_t frame_number, uint32_t partition_number, uint32_t data_reuse){
   uint32_t function_id = get_function_id(function_name);
   cnn_model* model = (cnn_model*)ctxt->model;
   uint32_t partitions_h = model->ftp_para->partitions_h;
   uint32_t partitions_w = model->ftp_para->partitions_w;
   uint32_t layers = model->ftp_para->fused_layers;
   if(deepthings_annot_data[partitions_h][partitions_w][layers][function_id].valid[frame_number][partition_number][data_reuse]){
      total_simulation_time = total_simulation_time + deepthings_annot_data[partitions_h][partitions_w][layers][function_id].avg_duration[frame_number][partition_number][data_reuse];
   }
   
}


