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
make ARGS="<Cli_ID> <wst/wst_s> <data_source/non_data_source>" test  
*/

/*
"models/yolo.cfg"
"models/yolo.weights"
*/

int main(int argc, char **argv){

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
   return 0;
}

