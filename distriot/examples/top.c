#include "darkiot.h"
#include "configure.h"
#include <string.h>
/*
#define DEBUG_FLAG 1

#define GATEWAY_PUBLIC_ADDR "10.157.89.51"
#define GATEWAY_LOCAL_ADDR "192.168.4.1"
#define EDGE_ADDR_LIST    {"192.168.4.9", "192.168.4.8", "192.168.4.4", "192.168.4.14", "192.168.4.15", "192.168.4.16"}
#define TOTAL_EDGE_NUM 6
#define FRAME_NUM 4
*/
/*The macro definitions*/
/*
make ARGS="2 start" test
make ARGS="2 wst gateway" test
make ARGS="0 wst data_source" test
make ARGS="1 wst non_data_source" test
make ARGS="2 wst non_data_source" test
*/

static const char* addr_list[MAX_EDGE_NUM] = EDGE_ADDR_LIST;

void test_gateway(uint32_t total_number){
   device_ctxt* ctxt = init_gateway(total_number, addr_list);
   set_gateway_local_addr(ctxt, GATEWAY_LOCAL_ADDR);
   set_gateway_public_addr(ctxt, GATEWAY_PUBLIC_ADDR);
   set_total_frames(ctxt, FRAME_NUM);
   set_batch_size(ctxt, BATCH_SIZE);

   sys_thread_t t3 = sys_thread_new("work_stealing_thread", work_stealing_thread, ctxt, 0, 0);
   sys_thread_t t1 = sys_thread_new("collect_result_thread", collect_result_thread, ctxt, 0, 0);
   sys_thread_t t2 = sys_thread_new("merge_result_thread", merge_result_thread, ctxt, 0, 0);
   exec_barrier(START_CTRL, TCP, ctxt);

   sys_thread_join(t1);
   sys_thread_join(t2);
   sys_thread_join(t3);


}

void test_stealer_client(uint32_t edge_id){
   device_ctxt* ctxt = init_client(edge_id);
   set_gateway_local_addr(ctxt, GATEWAY_LOCAL_ADDR);
   set_gateway_public_addr(ctxt, GATEWAY_PUBLIC_ADDR);
   set_total_frames(ctxt, FRAME_NUM);
   set_batch_size(ctxt, BATCH_SIZE);

   exec_barrier(START_CTRL, TCP, ctxt);

   sys_thread_t t1 = sys_thread_new("steal_and_process_thread", steal_and_process_thread, ctxt, 0, 0);
   sys_thread_t t2 = sys_thread_new("send_result_thread", send_result_thread, ctxt, 0, 0);

   sys_thread_join(t1);
   sys_thread_join(t2);

}

void test_victim_client(uint32_t edge_id){
   device_ctxt* ctxt = init_client(edge_id);
   set_gateway_local_addr(ctxt, GATEWAY_LOCAL_ADDR);
   set_gateway_public_addr(ctxt, GATEWAY_PUBLIC_ADDR);
   set_total_frames(ctxt, FRAME_NUM);
   set_batch_size(ctxt, BATCH_SIZE);

   exec_barrier(START_CTRL, TCP, ctxt);

   sys_thread_t t1 = sys_thread_new("generate_and_process_thread", generate_and_process_thread, ctxt, 0, 0);
   sys_thread_t t2 = sys_thread_new("send_result_thread", send_result_thread, ctxt, 0, 0);
   sys_thread_t t3 = sys_thread_new("serve_stealing_thread", serve_stealing_thread, ctxt, 0, 0);

   sys_thread_join(t1);
   sys_thread_join(t2);
   sys_thread_join(t3);


}

void test_wst(int argc, char **argv)
{

   printf("this_cli_id %d\n", atoi(argv[1]));

   uint32_t cli_id = atoi(argv[1]);
      
   if(0 == strcmp(argv[2], "start")){  
      printf("start\n");
      exec_start_gateway(START_CTRL, TCP, GATEWAY_PUBLIC_ADDR);
   }else if(0 == strcmp(argv[2], "wst")){
      printf("Work stealing\n");
      if(0 == strcmp(argv[3], "non_data_source")){
         printf("non_data_source\n");
         test_stealer_client(cli_id);
      }else if(0 == strcmp(argv[3], "data_source")){
         printf("data_source\n");
         test_victim_client(cli_id);
      }else if(0 == strcmp(argv[3], "gateway")){
         printf("gateway\n");
         test_gateway(cli_id);
      }
   }
}

int main(int argc, char **argv){
   /*test_queue_remove(argc, argv);*/
   test_wst(argc, argv);
   return 0;
}

