#ifndef TEST_UTIL_H
#define TEST_UTIL_H
#include "darkiot.h"
#include "configure.h"
#include "ftp.h"
#include "cmd_line_parser.h"
#include "frame_partitioner.h"
#include "deepthings_edge.h"
#include "deepthings_gateway.h"
#include "inference_engine_helper.h"
#include "frame_partitioner.h"
#include "reuse_data_serialization.h"
#include "deepthings_profile.h"

/*Functions defined for testing and profiling*/
void process_task_single_device(device_ctxt* edge_ctxt, device_ctxt* gateway_ctxt, blob* temp, bool is_reuse);
void process_everything_in_gateway(void *arg);
void transfer_data_with_number(device_ctxt* client, device_ctxt* gateway, int32_t task_num);
void transfer_data(device_ctxt* client, device_ctxt* gateway);
void deepthings_merge_result_thread_single_device(void *arg);
void partition_frame_and_perform_inference_thread_single_device(device_ctxt* edge_ctxt, device_ctxt* gateway_ctxt);
void partition_frame_and_perform_inference_thread_single_device_no_reuse(device_ctxt* edge_ctxt, device_ctxt* gateway_ctxt);

/*Functions defined for testing data reuse*/
bool* assume_all_are_missing(device_ctxt* ctxt, uint32_t task_id, uint32_t frame_num);
void request_reuse_data_single_device(device_ctxt* edge_ctxt, device_ctxt* gateway_ctxt, blob* task_input_blob, bool* reuse_data_is_required);
void send_reuse_data_single_device(device_ctxt* edge_ctxt, device_ctxt* gateway_ctxt, blob* task_input_blob);

#endif
