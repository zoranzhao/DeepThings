#include "deepthings_profile.h"

static deepthings_profile_data deepthings_prof_data[NUM_OF_FUNCTIONS];

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

static inline double now_sec(){
   struct timeval time;
   if (gettimeofday(&time,NULL)) return 0;
   return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

static inline double now_usec(){
   struct timeval time;
   if (gettimeofday(&time,NULL)) return 0;
   return (double)time.tv_sec * 1000000 + (double)time.tv_usec;
}

static inline uint32_t get_function_id(char* function_name){
   uint32_t id = 0; 
   for(id = 0; id < NUM_OF_FUNCTIONS; id++){
      if(strcmp(function_name, function_list[id]) == 0) return id;
   }
   return 0;
}

static inline char* get_function_name(uint32_t id){
   if(id < NUM_OF_FUNCTIONS) return function_list[id];
   return NULL;
}

void dump_profile(char* filename){
   FILE *f = fopen(filename, "w");
   if (f == NULL){
      printf("Error opening file!\n");
      exit(1);
   }
   uint32_t function_id;
   uint32_t frame_number;
   uint32_t partition_number;
   uint32_t data_reuse;
   fprintf(f, "name_of_function	frame#	partition#	reuse	calling#	avg_time\n");
   for(function_id = 0; function_id < NUM_OF_FUNCTIONS; function_id++){
      for(frame_number = 0; frame_number < FRAME_NUM; frame_number++){
         for(partition_number=0; partition_number<PARTITIONS_MAX; partition_number++){
            for(data_reuse=0; data_reuse<2; data_reuse++){
               if(deepthings_prof_data[function_id].valid[frame_number][partition_number][data_reuse] == false) continue;
               fprintf(f, "%s	%d	%d	%d	%ld	%f\n", 
                    get_function_name(function_id), 
                    frame_number, 
                    partition_number,
                    data_reuse,
                    deepthings_prof_data[function_id].calling_times[frame_number][partition_number][data_reuse], 
                    deepthings_prof_data[function_id].avg_duration[frame_number][partition_number][data_reuse]      
               );
            }
         }
      }
   }
   fclose(f);
}

void profile_start(){
   uint32_t function_id;
   uint32_t frame_number;
   uint32_t partition_number;
   for(function_id = 0; function_id < NUM_OF_FUNCTIONS; function_id++){
      for(frame_number = 0; frame_number < FRAME_NUM; frame_number++){
         for(partition_number=0; partition_number<PARTITIONS_MAX; partition_number++){
            deepthings_prof_data[function_id].valid[frame_number][partition_number][0] = false;
            deepthings_prof_data[function_id].valid[frame_number][partition_number][1] = false;
            deepthings_prof_data[function_id].total_duration[frame_number][partition_number][0] = 0.0;
            deepthings_prof_data[function_id].total_duration[frame_number][partition_number][1] = 0.0;
            deepthings_prof_data[function_id].avg_duration[frame_number][partition_number][0] = 0.0;
            deepthings_prof_data[function_id].avg_duration[frame_number][partition_number][1] = 0.0;
            deepthings_prof_data[function_id].calling_times[frame_number][partition_number][0] = 0;
            deepthings_prof_data[function_id].calling_times[frame_number][partition_number][1] = 0;
         }
      }
   }
}

void profile_end(uint32_t partition_h, uint32_t partition_w, uint32_t layers, uint32_t core){
   char filename[50];
   sprintf(filename, "%dx%d_grid_%d_layers_%d_core.prof", partition_h, partition_w, layers, core);
   dump_profile(filename);
}

void start_timer(char* function_name, uint32_t frame_number, uint32_t partition_number, uint32_t data_reuse){
   uint32_t function_id = get_function_id(function_name);
   deepthings_prof_data[function_id].start_time = now_usec();
}

void stop_timer(char* function_name, uint32_t frame_number, uint32_t partition_number, uint32_t data_reuse){
   uint32_t function_id = get_function_id(function_name);
   deepthings_prof_data[function_id].total_duration[frame_number][partition_number][data_reuse] += 
                          now_usec() - deepthings_prof_data[function_id].start_time;
   deepthings_prof_data[function_id].calling_times[frame_number][partition_number][data_reuse]++;
   deepthings_prof_data[function_id].avg_duration[frame_number][partition_number][data_reuse] =
                     deepthings_prof_data[function_id].total_duration[frame_number][partition_number][data_reuse]/
                      ((double)(deepthings_prof_data[function_id].calling_times[frame_number][partition_number][data_reuse]));
   deepthings_prof_data[function_id].valid[frame_number][partition_number][data_reuse] = true;
}


