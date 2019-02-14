#ifndef NETWORK_UTIL_H
#define NETWORK_UTIL_H
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <stdio.h>
#include <sys/types.h> 
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <errno.h>
/*Assgin port number for different services*/
#define PORTNO 11111 //Service for job stealing and sharing
#define SMART_GATEWAY 11112 //Service for a smart gateway 
#define START_CTRL 11113 //Control the start and stop of a service
#define RESULT_COLLECT_PORT 11114 //Control the start and stop of a service
#define WORK_STEAL_PORT 11115 //Control the start and stop of a service

#define IPV4_TASK 1
#define IPV6_TASK !(IPV4_TASK)

#if IPV4_TASK
#define ADDRSTRLEN INET_ADDRSTRLEN
#elif IPV6_TASK/*IPV4_TASK*/
#define ADDRSTRLEN INET6_ADDRSTRLEN
#endif/*IPV4_TASK*/   

#include "data_blob.h"

typedef enum proto{
   TCP,
   UDP
} ctrl_proto;

typedef struct service_connection{
   int sockfd;
   ctrl_proto proto;
   #if IPV4_TASK
   struct sockaddr_in* serv_addr_ptr;
   #elif IPV6_TASK/*IPV4_TASK*/
   struct sockaddr_in6* serv_addr_ptr;
   #endif/*IPV4_TASK*/   
} service_conn;

/*Networking API on service client side*/
service_conn* connect_service(ctrl_proto proto, const char *dest_ip, int portno);
void close_service_connection(service_conn* conn);
void send_request(char* meta, uint32_t meta_size, service_conn* conn);

/*Networking API on service server side*/
int service_init(int portno, ctrl_proto proto);
void start_service_for_n_times(int sockfd, ctrl_proto proto, const char* handler_name[], uint32_t handler_num, void* (*handlers[])(void*, void*), void* arg, uint32_t times);
void start_service(int sockfd, ctrl_proto proto, const char* handler_name[], uint32_t handler_num, void* (*handlers[])(void*, void*), void* arg);
void close_service(int sockfd);

/*Data exchanging API on both sides*/
blob* recv_data(service_conn* conn);
void send_data(blob *temp, service_conn* conn);

/*IP address parsing API*/
void get_dest_ip_string(char* ip_string, service_conn* conn);

#endif
