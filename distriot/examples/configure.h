#ifndef CONFIG_H
#define CONFIG_H
#include <stdint.h>

/*Debug printing information*/

#define GATEWAY_PUBLIC_ADDR "10.157.89.51"
#define GATEWAY_LOCAL_ADDR "192.168.4.1"
#define EDGE_ADDR_LIST    {"192.168.4.9", "192.168.4.8", "192.168.4.4", "192.168.4.14", "192.168.4.15", "192.168.4.16"}
#define MAX_EDGE_NUM 6
#define FRAME_NUM 4
#define BATCH_SIZE 16

/*IP address*/
/*
#define AP "10.157.89.51"
#define GATEWAY "192.168.4.1"
*/
/*
#define BLUE1    "192.168.4.9"
#define ORANGE1  "192.168.4.8"
#define PINK1    "192.168.4.4"
#define BLUE0    "192.168.4.14"
#define ORANGE0  "192.168.4.15"
#define PINK0    "192.168.4.16"
#define EDGE_ADDR_LIST    {"192.168.4.9", "192.168.4.8", "192.168.4.4", "192.168.4.14", "192.168.4.15", "192.168.4.16"}
#define EDGE_ID_LIST    {0, 1, 2, 3, 4, 5}
#define TOTAL_EDGE_NUM 6
*/
/*Client number...*/
/*#define CLI_NUM 6*/
/*Global configurations defined in config.c*/
/*
extern const char* addr_list[CLI_NUM];
extern uint32_t this_cli_id;
extern uint32_t total_cli_num;
*/
/*Other parameters*/
 
/*
#define FRAME_NUM 4
*/
#endif
