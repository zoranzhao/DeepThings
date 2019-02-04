#ifndef GATEWAY_H
#define GATEWAY_H
#include "darkiot.h"

#ifdef __cplusplus
extern "C" {
#endif
device_ctxt* init_gateway(uint32_t cli_num, const char** edge_addr_list);
void collect_result_thread(void *arg);
void merge_result_thread(void *arg);
void work_stealing_thread(void *arg);
void* register_gateway(void* srv_conn, void *arg);
void* cancel_gateway(void* srv_conn, void *arg);
void* steal_gateway(void* srv_conn, void *arg);
#ifdef __cplusplus
}
#endif

#endif
