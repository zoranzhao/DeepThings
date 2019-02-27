#include "test_utils.h"
image **load_alphabet_local_data(){
    int i, j;
    const int nsize = 8;
    image **alphabets = calloc(nsize, sizeof(image));
    for(j = 0; j < nsize; ++j){
        alphabets[j] = calloc(128, sizeof(image));
        for(i = 32; i < 127; ++i){
            char buff[256];
            sprintf(buff, "../data/labels/%d_%d.png", i, j);
            alphabets[j][i] = load_image_color(buff, 0, 0);
        }
    }
    return alphabets;
}



void load_image_by_number_local_data(image* img, uint32_t id){
   int32_t h = img->h;
   int32_t w = img->w;
   char filename[256];
   sprintf(filename, "../data/input/%d.jpg", id);
   image im = load_image_color(filename, 0, 0);
   image sized = letterbox_image(im, w, h);
   free_image(im);
   img->data = sized.data;
}

image load_image_as_model_input_local_data(cnn_model* model, uint32_t id){
   image sized;
   sized.w = model->net->w; 
   sized.h = model->net->h; 
   sized.c = model->net->c;
   load_image_by_number_local_data(&sized, id);
   model->net->input = sized.data;
   return sized;
}

void draw_object_boxes_local_data(cnn_model* model, uint32_t id){
   network net = *(model->net);
   image sized;
   sized.w = net.w; sized.h = net.h; sized.c = net.c;
   load_image_by_number_local_data(&sized, id);
   image **alphabet = load_alphabet_local_data();
   list *options = read_data_cfg((char*)"./data/coco.data");
   char *name_list = option_find_str(options, (char*)"names", (char*)"./data/names.list");
   printf("%s\n =====\n", name_list);
   char **names = get_labels(name_list);
   char filename[256];
   char outfile[256];
   float thresh = .24;
   float hier_thresh = .5;
   float nms=.3;
   sprintf(filename, "../data/input/%d.jpg", id);
   sprintf(outfile, "%d", id);
   layer l = net.layers[net.n-1];
   float **masks = 0;
   if (l.coords > 4){
      masks = (float **)calloc(l.w*l.h*l.n, sizeof(float*));
      for(int j = 0; j < l.w*l.h*l.n; ++j) masks[j] = (float *)calloc(l.coords-4, sizeof(float *));
   }
   float **probs = (float **)calloc(l.w*l.h*l.n, sizeof(float *));
   for(int j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float *)calloc(l.classes + 1, sizeof(float *));
   image im = load_image_color(filename,0,0);
   box *boxes = (box *)calloc(l.w*l.h*l.n, sizeof(box));
   get_region_boxes(l, im.w, im.h, net.w, net.h, thresh, probs, boxes, masks, 0, 0, hier_thresh, 1);
   if (nms) do_nms_sort(boxes, probs, l.w*l.h*l.n, l.classes, nms);
   draw_detections(im, l.w*l.h*l.n, thresh, boxes, probs, masks, names, alphabet, l.classes);
   save_image(im, outfile);
   free(boxes);
   free_ptrs((void **)probs, l.w*l.h*l.n);
   if (l.coords > 4){
      free_ptrs((void **)masks, l.w*l.h*l.n);
   }
   free_image(im);
}

void process_everything_in_gateway(void *arg){
   cnn_model* model = (cnn_model*)(((device_ctxt*)(arg))->model);
#ifdef NNPACK
   nnp_initialize();
   model->net->threadpool = pthreadpool_create(THREAD_NUM);
#endif
   int32_t frame_num;
   for(frame_num = 0; frame_num < FRAME_NUM; frame_num ++){
      image_holder img = load_image_as_model_input_local_data(model, frame_num);
      forward_all(model, 0);   
      draw_object_boxes_local_data(model, frame_num);
      free_image_holder(model, img);
   }
#ifdef NNPACK
   pthreadpool_destroy(model->net->threadpool);
   nnp_deinitialize();
#endif
}

void process_task_single_device(device_ctxt* edge_ctxt, device_ctxt* gateway_ctxt, blob* temp, bool is_reuse){
   printf("Task is: %d, frame number is %d\n", get_blob_task_id(temp), get_blob_frame_seq(temp));
   cnn_model* model = (cnn_model*)(edge_ctxt->model);
   blob* result;
   set_model_input(model, (float*)temp->data);
   start_timer("forward_partition", get_blob_frame_seq(temp), get_blob_task_id(temp), is_reuse);
   forward_partition(model, get_blob_task_id(temp), is_reuse);  
   stop_timer("forward_partition", get_blob_frame_seq(temp), get_blob_task_id(temp), is_reuse);
   result = new_blob_and_copy_data(0, 
                                      get_model_byte_size(model, model->ftp_para->fused_layers-1), 
                                      (uint8_t*)(get_model_output(model, model->ftp_para->fused_layers-1))
                                     );
#if DATA_REUSE
   /*send_reuse_data(ctxt, temp);*/
   /*if task doesn't generate any reuse_data*/
   set_coverage(model->ftp_para_reuse, get_blob_task_id(temp), get_blob_frame_seq(temp));
   send_reuse_data_single_device(edge_ctxt, gateway_ctxt, temp);
#endif

   copy_blob_meta(result, temp);
   enqueue(edge_ctxt->result_queue, result); 
   free_blob(result);
}

void partition_frame_and_perform_inference_thread_single_device(device_ctxt* edge_ctxt, device_ctxt* gateway_ctxt){
   cnn_model* model = (cnn_model*)(edge_ctxt->model);
#ifdef NNPACK
   nnp_initialize();
   model->net->threadpool = pthreadpool_create(THREAD_NUM);
#endif
   blob* temp;
   uint32_t frame_num;
   /*bool* reuse_data_is_required;*/   
   for(frame_num = 0; frame_num < FRAME_NUM; frame_num ++){
      /*Wait for i/o device input*/
      /*recv_img()*/

      /*Load image and partition, fill task queues*/
      start_timer("load_image_as_model_input", frame_num, 0, 0);
      load_image_as_model_input_local_data(model, frame_num);
      stop_timer("load_image_as_model_input", frame_num, 0, 0);

      start_timer("partition_and_enqueue", frame_num, 0, 0);
      partition_and_enqueue(edge_ctxt, frame_num);
      start_timer("partition_and_enqueue", frame_num, 0, 0);
      /*register_client(edge_ctxt);*/

      /*Dequeue and process task*/
      while(1){
         temp = try_dequeue(edge_ctxt->task_queue);
         if(temp == NULL) break;
         bool data_ready = false;
         printf("====================Processing task id is %d, data source is %d, frame_seq is %d====================\n", get_blob_task_id(temp), get_blob_cli_id(temp), get_blob_frame_seq(temp));
#if DATA_REUSE
         data_ready = is_reuse_ready(model->ftp_para_reuse, get_blob_task_id(temp), frame_num);
         if((model->ftp_para_reuse->schedule[get_blob_task_id(temp)] == 1) && data_ready) {
            blob* shrinked_temp = new_blob_and_copy_data(get_blob_task_id(temp), 
                       (model->ftp_para_reuse->shrinked_input_size[get_blob_task_id(temp)]),
                       (uint8_t*)(model->ftp_para_reuse->shrinked_input[get_blob_task_id(temp)]));
            copy_blob_meta(shrinked_temp, temp);
            free_blob(temp);
            temp = shrinked_temp;

            /*If we do nothing here, we are assuming that all reusable data is generated locally*/
            /*
            reuse_data_is_required = check_missing_coverage(model, get_blob_task_id(temp), get_blob_frame_seq(temp));
            request_reuse_data(edge_ctxt, temp, reuse_data_is_required);
            free(reuse_data_is_required);
	    */

            /*We are now assuming all data is missing, we need to grab everything from gateway*/
            bool* reuse_data_is_required = assume_all_are_missing(edge_ctxt, get_blob_task_id(temp), get_blob_frame_seq(temp));
            request_reuse_data_single_device(edge_ctxt, gateway_ctxt, temp, reuse_data_is_required);
         }
#if DEBUG_DEEP_EDGE
         if((model->ftp_para_reuse->schedule[get_blob_task_id(temp)] == 1) && (!data_ready))
            printf("The reuse data is not ready yet!\n");
#endif/*DEBUG_DEEP_EDGE*/

#endif/*DATA_REUSE*/
         /*process_task(edge_ctxt, temp, data_ready);*/
         process_task_single_device(edge_ctxt, gateway_ctxt, temp, data_ready);
         free_blob(temp);
      }

      /*Unregister and prepare for next image*/
      /*cancel_client(edge_ctxt);*/
   }
#ifdef NNPACK
   pthreadpool_destroy(model->net->threadpool);
   nnp_deinitialize();
#endif
}


void partition_frame_and_perform_inference_thread_single_device_no_reuse(device_ctxt* edge_ctxt, device_ctxt* gateway_ctxt){
   cnn_model* model = (cnn_model*)(edge_ctxt->model);
#ifdef NNPACK
   nnp_initialize();
   model->net->threadpool = pthreadpool_create(THREAD_NUM);
#endif
   blob* temp;
   uint32_t frame_num;
   /*bool* reuse_data_is_required;*/   
   for(frame_num = 0; frame_num < FRAME_NUM; frame_num ++){
      /*Wait for i/o device input*/
      /*recv_img()*/

      /*Load image and partition, fill task queues*/
      start_timer("load_image_as_model_input", frame_num, 0, 0);
      load_image_as_model_input_local_data(model, frame_num);
      stop_timer("load_image_as_model_input", frame_num, 0, 0);

      start_timer("partition_and_enqueue", frame_num, 0, 0);
      partition_and_enqueue(edge_ctxt, frame_num);
      start_timer("partition_and_enqueue", frame_num, 0, 0);
      /*register_client(edge_ctxt);*/

      /*Dequeue and process task*/
      while(1){
         temp = try_dequeue(edge_ctxt->task_queue);
         if(temp == NULL) break;
         bool data_ready = false;
         printf("====================Processing task id is %d, data source is %d, frame_seq is %d====================\n", get_blob_task_id(temp), get_blob_cli_id(temp), get_blob_frame_seq(temp));
         /*process_task(edge_ctxt, temp, data_ready);*/
         process_task_single_device(edge_ctxt, gateway_ctxt, temp, data_ready);
         free_blob(temp);
      }
   }
#ifdef NNPACK
   pthreadpool_destroy(model->net->threadpool);
   nnp_deinitialize();
#endif
}

void deepthings_merge_result_thread_single_device(void *arg){
   cnn_model* model = (cnn_model*)(((device_ctxt*)(arg))->model);
#ifdef NNPACK
   nnp_initialize();
   model->net->threadpool = pthreadpool_create(THREAD_NUM);
#endif
   blob* temp;
   int32_t cli_id;
   int32_t frame_seq;
   int32_t count = 0;
   for(count = 0; count < FRAME_NUM; count ++){
      temp = dequeue_and_merge((device_ctxt*)arg);
      cli_id = get_blob_cli_id(temp);
      frame_seq = get_blob_frame_seq(temp);
#if DEBUG_FLAG
      printf("Client %d, frame sequence number %d, all partitions are merged in deepthings_merge_result_thread\n", cli_id, frame_seq);
#endif
      float* fused_output = (float*)(temp->data);
      image_holder img = load_image_as_model_input_local_data(model, get_blob_frame_seq(temp));
      set_model_input(model, fused_output);
      forward_all(model, model->ftp_para->fused_layers);   
      draw_object_boxes_local_data(model, get_blob_frame_seq(temp));
      free_image_holder(model, img);
      free_blob(temp);
#if DEBUG_FLAG
      printf("Client %d, frame sequence number %d, finish processing\n", cli_id, frame_seq);
#endif
   }
#ifdef NNPACK
   pthreadpool_destroy(model->net->threadpool);
   nnp_deinitialize();
#endif
}

void transfer_data(device_ctxt* client, device_ctxt* gateway){
   int32_t cli_id = client->this_cli_id;
   uint32_t frame_seq = 0;
   while(1){
      blob* temp = dequeue(client->result_queue);
      printf("Transfering data from client %d to gateway\n", cli_id);
      enqueue(gateway->results_pool[cli_id], temp);
      gateway->results_counter[cli_id]++;
      free_blob(temp);
      if(gateway->results_counter[cli_id] == gateway->batch_size){
         temp = new_empty_blob(cli_id);
         annotate_blob(temp, cli_id, frame_seq, 0);
         frame_seq++;
         enqueue(gateway->ready_pool, temp);
         free_blob(temp);
         gateway->results_counter[cli_id] = 0;
      }
   }
}

void transfer_data_with_number(device_ctxt* client, device_ctxt* gateway, int32_t task_num){
   int32_t cli_id = client->this_cli_id;
   int32_t count = 0;
   uint32_t frame_seq = 0;
   for(count = 0; count < task_num; count ++){
      blob* temp = dequeue(client->result_queue);
      printf("Transfering data from client %d to gateway\n", cli_id);
      enqueue(gateway->results_pool[cli_id], temp);
      gateway->results_counter[cli_id]++;
      free_blob(temp);
      if(gateway->results_counter[cli_id] == gateway->batch_size){
         temp = new_empty_blob(cli_id);
         annotate_blob(temp, cli_id, frame_seq, 0);
         frame_seq++;
         enqueue(gateway->ready_pool, temp);
         free_blob(temp);
         gateway->results_counter[cli_id] = 0;
      }
   }
}


/*-------------------------------------------------------*/
//All functions need to be invoked in client device
bool* assume_all_are_missing(device_ctxt* ctxt, uint32_t task_id, uint32_t frame_num){
   cnn_model* model = (cnn_model*)(ctxt->model);
   uint32_t pos;
   bool* reuse_data_is_required = (bool*) malloc(4*sizeof(bool));
   int32_t* adjacent_id = get_adjacent_task_id_list(model, task_id);
   for(pos = 0; pos < 4; pos++){
      reuse_data_is_required[pos] = true;
      if(adjacent_id[pos]==-1){
         reuse_data_is_required[pos] = false;
      }
   }
   free(adjacent_id);
   return reuse_data_is_required;
}

/*-------------------------------------------------------*/

/*Reuse data fetching stage*/
//Hand out reuse data to edge devices
static overlapped_tile_data* overlapped_data_pool[MAX_EDGE_NUM][PARTITIONS_MAX];
blob* send_reuse_data_to_edge_single_device(device_ctxt* edge_ctxt, device_ctxt* gateway_ctxt, blob* recv_meta, blob* recv_missing_info){
   cnn_model* gateway_model = (cnn_model*)(gateway_ctxt->model);
   int32_t cli_id;
   int32_t task_id;
   uint32_t frame_num;
   cli_id = get_blob_cli_id(recv_meta);
   task_id = get_blob_task_id(recv_meta);
   frame_num = get_blob_frame_seq(recv_meta);
   free_blob(recv_meta);

   bool* reuse_data_is_required = (bool*)(recv_missing_info->data);
   int32_t* adjacent_id = get_adjacent_task_id_list(gateway_model, task_id);
   uint32_t position;
   for(position = 0; position < 4; position++){
      if(adjacent_id[position]==-1) continue;
      if(reuse_data_is_required[position]){
         printf("In gateway, collect reuse data generated from task %d\n", adjacent_id[position]);
         start_timer("place_self_deserialized_data", frame_num, cli_id, 1);
         place_self_deserialized_data(gateway_model, adjacent_id[position], overlapped_data_pool[cli_id][adjacent_id[position]]);
         stop_timer("place_self_deserialized_data", frame_num, cli_id, 1);
      }
   }

   free(adjacent_id);
   start_timer("adjacent_reuse_data_serialization", frame_num, cli_id, 1);
   blob* temp = adjacent_reuse_data_serialization(gateway_ctxt, task_id, frame_num, reuse_data_is_required);
   stop_timer("adjacent_reuse_data_serialization", frame_num, cli_id, 1);
   free_blob(recv_missing_info);
   return temp;

}

//Request reuse data from gateway
void request_reuse_data_single_device(device_ctxt* edge_ctxt, device_ctxt* gateway_ctxt, blob* task_input_blob, bool* reuse_data_is_required){
   cnn_model* model = (cnn_model*)(edge_ctxt->model);
   if(model->ftp_para_reuse->schedule[get_blob_task_id(task_input_blob)] == 0) return;
   //Need reuse data anyway fior profiling
   //if(!need_reuse_data_from_gateway(reuse_data_is_required)) return;

   //Meta data including task_id, frame_number and client_id
   char data[20]="empty";
   blob* temp = new_blob_and_copy_data(get_blob_task_id(task_input_blob), 20, (uint8_t*)data);
   copy_blob_meta(temp, task_input_blob);
   blob* recv_meta = temp;
   //send_data(temp, conn);
   //free_blob(temp);

   //Meta data including the locations for missing part of reuse data  
   temp = new_blob_and_copy_data(get_blob_task_id(task_input_blob), sizeof(bool)*4, (uint8_t*)reuse_data_is_required);
   copy_blob_meta(temp, task_input_blob);
   blob* recv_missing_info = temp;
   //send_data(temp, conn);
   //free_blob(temp);

   //receive reuse data 
   //temp = recv_data(conn);
   temp = send_reuse_data_to_edge_single_device(edge_ctxt, gateway_ctxt, recv_meta, recv_missing_info);
   copy_blob_meta(temp, task_input_blob);
   start_timer("adjacent_reuse_data_deserialization", get_blob_frame_seq(temp), get_blob_task_id(temp), 1);
   overlapped_tile_data** temp_region_and_data = adjacent_reuse_data_deserialization(model, get_blob_task_id(temp), (float*)temp->data, get_blob_frame_seq(temp), reuse_data_is_required);
   stop_timer("adjacent_reuse_data_deserialization", get_blob_frame_seq(temp), get_blob_task_id(temp), 1);
   start_timer("place_adjacent_deserialized_data", get_blob_frame_seq(temp), get_blob_task_id(temp), 1);
   place_adjacent_deserialized_data(model, get_blob_task_id(temp), temp_region_and_data, reuse_data_is_required);
   stop_timer("place_adjacent_deserialized_data", get_blob_frame_seq(temp), get_blob_task_id(temp), 1);
   free_blob(temp);
}




/*Reuse data collection stage*/
//Collect reuse data from edge devices
void recv_reuse_data_from_edge_single_device(device_ctxt* gateway, blob* recv_data_blob){
   cnn_model* gateway_model = (cnn_model*)gateway->model;
   blob* temp = recv_data_blob;
   int32_t cli_id = 0;
   int32_t task_id = get_blob_task_id(temp);
   printf("collecting_reuse_data generated by task %d, client %d... ... \n", task_id, cli_id);
   if(overlapped_data_pool[cli_id][task_id] != NULL)
      free_self_overlapped_tile_data(gateway_model,  overlapped_data_pool[cli_id][task_id]);
   start_timer("self_reuse_data_deserialization", get_blob_frame_seq(temp), get_blob_task_id(temp), 1);
   overlapped_data_pool[cli_id][task_id] = self_reuse_data_deserialization(gateway_model, task_id, (float*)temp->data, get_blob_frame_seq(temp));
   stop_timer("self_reuse_data_deserialization", get_blob_frame_seq(temp), get_blob_task_id(temp), 1);
   //Should I?//free_blob(temp);
}

//Send generated reuse data to gateway 
void send_reuse_data_single_device(device_ctxt* edge_ctxt, device_ctxt* gateway_ctxt, blob* task_input_blob){
   cnn_model* model = (cnn_model*)(edge_ctxt->model);
   if(model->ftp_para_reuse->schedule[get_blob_task_id(task_input_blob)] == 1) return;

   start_timer("self_reuse_data_serialization", get_blob_frame_seq(task_input_blob), get_blob_task_id(task_input_blob), 1);
   blob* temp  = self_reuse_data_serialization(edge_ctxt, get_blob_task_id(task_input_blob), get_blob_frame_seq(task_input_blob));
   stop_timer("self_reuse_data_serialization", get_blob_frame_seq(task_input_blob), get_blob_task_id(task_input_blob), 1);

   copy_blob_meta(temp, task_input_blob);
   //Transfer it to the gateway  
   recv_reuse_data_from_edge_single_device(gateway_ctxt, temp);
   //Transfer it to the gateway 
   free_blob(temp);
}


