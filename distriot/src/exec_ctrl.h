#ifndef EXEC_CTRL_H
#define EXEC_CTRL_H
#include "data_blob.h"
#include "thread_util.h"
#include "thread_safe_queue.h"
#include "network_util.h"
#include "config.h"

void exec_barrier(int portno, ctrl_proto proto);
void exec_start_gateway(int portno, ctrl_proto proto);
int32_t get_client_id(const char* ip_addr);
char* get_client_addr(int32_t cli_id);

int32_t get_this_client_id();
void annotate_blob(blob* temp, int32_t cli_id, int32_t frame_seq, int32_t task_id);
int32_t get_blob_cli_id(blob* temp);
int32_t get_blob_frame_seq(blob* temp);
int32_t get_blob_task_id(blob* temp);

#endif
