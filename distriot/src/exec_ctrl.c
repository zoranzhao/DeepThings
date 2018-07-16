#include "exec_ctrl.h"

void exec_start_gateway(int portno, ctrl_proto proto, char* gateway_public_addr){
    char request_type[20] = "start_gateway";
    service_conn* conn = connect_service(proto, gateway_public_addr, portno);
    send_request(request_type, 20, conn);
    close_service_connection(conn);
}

void* start_gateway(void* conn, void* arg){
   char request_type[20] = "start_edge";
   printf("Call start_gateway, start edge devices ...\n");
   uint32_t cli_id;
   service_conn* new_conn;
   for(cli_id = 0; cli_id < ((device_ctxt*)arg)->total_cli_num; cli_id ++){
      new_conn = connect_service(((service_conn*)conn)->proto, ((device_ctxt*)arg)->addr_list[cli_id], START_CTRL);
      send_request(request_type, 20, new_conn);
      close_service_connection(new_conn);
   }
   return NULL;
}

void* start_edge(void* conn, void* arg){
   printf("Call start_edge, Begin to do the work ...\n");
   return NULL;
}

void exec_barrier(int portno, ctrl_proto proto, device_ctxt* ctxt)
{
   const char* request_types[]={"start_gateway", "start_edge"};
   void* (*handlers[])(void*, void*) = {start_gateway, start_edge};
   int service = service_init(portno, proto);
   start_service_for_n_times(service, proto, request_types, 2, handlers, ctxt, 1);
   close_service(service);
}

int32_t get_client_id(const char* ip_addr, device_ctxt* ctxt){
   uint32_t i;
   for(i = 0; i < ctxt->total_cli_num; i++){
      if(strcmp(ip_addr, ctxt->addr_list[i]) == 0){return i;}
   }
   return (-1);//
}

const char* get_client_addr(int32_t cli_id, device_ctxt* ctxt){
   return ctxt->addr_list[cli_id];
}

int32_t get_this_client_id(device_ctxt* ctxt){
   return ctxt->this_cli_id;
}


void annotate_blob(blob* temp, int32_t cli_id, int32_t frame_seq, int32_t task_id){
   int32_t meta[3];
   meta[0] = cli_id;
   meta[1] = frame_seq;
   meta[2] = task_id;
   fill_blob_meta(temp, sizeof(int32_t)*3, (uint8_t*)meta);
}

int32_t get_blob_cli_id(blob* temp){
   int32_t *meta = (int32_t*)(temp->meta);
   return meta[0];
}

int32_t get_blob_frame_seq(blob* temp){
   int32_t *meta = (int32_t*)(temp->meta);
   return meta[1];
}

int32_t get_blob_task_id(blob* temp){
   int32_t *meta = (int32_t*)(temp->meta);
   return meta[2];
}



