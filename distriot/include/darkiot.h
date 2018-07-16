#ifndef DARKIOT_H
#define DARKIOT_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include <unistd.h>
#include <sys/types.h> 
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <time.h>

#include "thread_safe_queue.h"
#include "thread_util.h"
#include "data_blob.h"
#include "network_util.h"
#include "exec_ctrl.h"
#include "client.h"
#include "gateway.h"
#include "global_context.h"

#define DEBUG_FLAG 1

#endif


