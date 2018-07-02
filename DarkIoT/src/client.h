#ifndef CLIENT_H
#define CLIENT_H
#include "config.h"
#include "exec_ctrl.h"

void init_client();
void steal_and_process_thread(void *arg);
void generate_and_process_thread(void *arg);
void send_result_thread(void *arg);
void serve_stealing_thread(void *arg);
void register_client();
void cancel_client();
void* steal_client(void* srv_conn);

#endif
