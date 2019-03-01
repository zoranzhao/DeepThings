#include "test_utils.h"

static const char* addr_list[MAX_EDGE_NUM] = EDGE_ADDR_LIST;


int main(int argc, char **argv){
   /*Initialize the data structure and network model*/
   uint32_t total_cli_num = get_int_arg(argc, argv, "-total_edge", 1);
   uint32_t this_cli_id = get_int_arg(argc, argv, "-edge_id", 0);

   uint32_t partitions_h = get_int_arg(argc, argv, "-n", 5);
   uint32_t partitions_w = get_int_arg(argc, argv, "-m", 5);
   uint32_t fused_layers = get_int_arg(argc, argv, "-l", 16);

   char network_file[30] = "../../models/yolo.cfg";
   char weight_file[30] = "../../models/yolo.weights";

   /*Initialize the data structures to be used in profiling*/
   profile_start();

   device_ctxt* client_ctxt = deepthings_edge_init(partitions_h, partitions_w, fused_layers, network_file, weight_file, this_cli_id);
   device_ctxt* gateway_ctxt = deepthings_gateway_init(partitions_h, partitions_w, fused_layers, network_file, weight_file, total_cli_num, addr_list);

   /*Profile with data-reuse execution*/
   partition_frame_and_perform_inference_thread_single_device(client_ctxt, gateway_ctxt);
   transfer_data_with_number(client_ctxt, gateway_ctxt, partitions_h*partitions_w*FRAME_NUM);
   deepthings_merge_result_thread_single_device(gateway_ctxt);

   /*Profile without data-reuse execution*/
   partition_frame_and_perform_inference_thread_single_device_no_reuse(client_ctxt, gateway_ctxt);
   transfer_data_with_number(client_ctxt, gateway_ctxt, partitions_h*partitions_w*FRAME_NUM);
   deepthings_merge_result_thread_single_device(gateway_ctxt);

   profile_end(partitions_h, partitions_w, fused_layers, THREAD_NUM);

   return 0;
}

