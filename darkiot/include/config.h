#ifndef CONFIG_H
#define CONFIG_H
#include <stdint.h>

/*Debug printing information*/
#define DEBUG_FLAG 1

/*Assgin port number for different services*/
#define PORTNO 11111 //Service for job stealing and sharing
#define SMART_GATEWAY 11112 //Service for a smart gateway 
#define START_CTRL 11113 //Control the start and stop of a service
#define RESULT_COLLECT_PORT 11114 //Control the start and stop of a service
#define WORK_STEAL_PORT 11115 //Control the start and stop of a service

/*IP address*/
#define AP "10.157.89.51"/*Local ip is "192.168.4.1"*/
#define GATEWAY "192.168.4.1"
#define BLUE1    "192.168.4.9"
#define ORANGE1  "192.168.4.8"
#define PINK1    "192.168.4.4"
#define BLUE0    "192.168.4.14"
#define ORANGE0  "192.168.4.15"
#define PINK0    "192.168.4.16"

/*Client number...*/
#define CLI_NUM 6
/*Global configurations defined in config.c*/
extern char* addr_list[CLI_NUM];
extern uint32_t this_cli_id;
extern uint32_t total_cli_num;

/*Other parameters*/
#define MAX_QUEUE_SIZE 256 
#define BATCH_SIZE 16 /* Number of elements gateway should merge at*/
#define FRAME_NUM 1

#endif
