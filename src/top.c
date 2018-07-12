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
./deepthings start
./deepthings gateway 6
./deepthings data_src 0
./deepthings non_data_src 1
*/

/*
"models/yolo.cfg", "models/yolo.weights"
*/

int main(int argc, char **argv){
   total_cli_num = 0;
   this_cli_id = 0;

   uint32_t partitions_h = 5;
   uint32_t partitions_w = 5;
   uint32_t fused_layers = 16;

   char network_file[30] = "models/yolo.cfg";
   char weight_file[30] = "models/yolo.weights";

   if(0 == strcmp(argv[1], "start")){  
      printf("start\n");
      exec_start_gateway(START_CTRL, TCP);
   }else if(0 == strcmp(argv[1], "gateway")){
      printf("Gateway device\n");
      printf("We have %d edge devices now\n", atoi(argv[2]));
      total_cli_num = atoi(argv[2]);
      deepthings_gateway(partitions_h, partitions_w, fused_layers, network_file, weight_file);
   }else if(0 == strcmp(argv[1], "data_src")){
      printf("Data source edge device\n");
      printf("This client ID is %d\n", atoi(argv[2]));
      this_cli_id = atoi(argv[2]);
      deepthings_victim_edge(partitions_h, partitions_w, fused_layers, network_file, weight_file);
   }else if(0 == strcmp(argv[1], "non_data_src")){
      printf("Idle edge device\n");
      printf("This client ID is %d\n", atoi(argv[2]));
      this_cli_id = atoi(argv[2]);
      deepthings_stealer_edge(partitions_h, partitions_w, fused_layers, network_file, weight_file);
   }
   return 0;
}

