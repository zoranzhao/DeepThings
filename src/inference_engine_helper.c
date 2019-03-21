#include "inference_engine_helper.h"

cnn_model* load_cnn_model(char* cfg, char* weights){
   cnn_model* model = (cnn_model*)malloc(sizeof(cnn_model));
   network *net = load_network(cfg, weights, 0);
   set_batch_network(net, 1);
   net->truth = 0;
   net->train = 0;
   net->delta = 0;
   srand(2222222);
   model->net = net;
   /*Extract and record network parameters*/
   model->net_para = (network_parameters*)malloc(sizeof(network_parameters));
   model->net_para->layers= net->n;   
   model->net_para->stride = (uint32_t*)malloc(sizeof(uint32_t)*(net->n));
   model->net_para->filter = (uint32_t*)malloc(sizeof(uint32_t)*(net->n));
   model->net_para->type = (uint32_t*)malloc(sizeof(uint32_t)*(net->n));
   model->net_para->input_maps = (tile_region*) malloc(sizeof(tile_region)*(net->n));
   model->net_para->output_maps = (tile_region*) malloc(sizeof(tile_region)*(net->n));
   uint32_t l;
   for(l = 0; l < (net->n); l++){
      model->net_para->stride[l] = net->layers[l].stride;
      model->net_para->filter[l] = net->layers[l].size;
      model->net_para->type[l] = net->layers[l].type;
      model->net_para->input_maps[l].w1 = 0;
      model->net_para->input_maps[l].h1 = 0;
      model->net_para->input_maps[l].w2 = net->layers[l].w - 1;
      model->net_para->input_maps[l].h2 = net->layers[l].h - 1;
      model->net_para->input_maps[l].w = net->layers[l].w;
      model->net_para->input_maps[l].h = net->layers[l].h;
      model->net_para->input_maps[l].c = net->layers[l].c;

      model->net_para->output_maps[l].w1 = 0;
      model->net_para->output_maps[l].h1 = 0;
      model->net_para->output_maps[l].w2 = net->layers[l].out_w - 1;
      model->net_para->output_maps[l].h2 = net->layers[l].out_h - 1;
      model->net_para->output_maps[l].w = net->layers[l].out_w;
      model->net_para->output_maps[l].h = net->layers[l].out_h;
      model->net_para->output_maps[l].c = net->layers[l].out_c;
   }
   return model;
}

/*input(w*h) [dh1, dh2]    copy into ==> output  [0, dh2 - dh1]
	     [dw1, dw2]			         [0, dw2 - dw1]*/
float* crop_feature_maps(float* input, uint32_t w, uint32_t h, uint32_t c, uint32_t dw1, uint32_t dw2, uint32_t dh1, uint32_t dh2){
   uint32_t out_w = dw2 - dw1 + 1;
   uint32_t out_h = dh2 - dh1 + 1;
   uint32_t i,j,k;
   uint32_t in_index;
   uint32_t out_index;
   float* output = (float*) malloc( sizeof(float)*out_w*out_h*c );  
   for(k = 0; k < c; ++k){
      for(j = dh1; j < dh2+1; ++j){
         for(i = dw1; i < dw2+1; ++i){
            in_index  = i + w*(j + h*k);
            out_index  = (i - dw1) + out_w*(j - dh1) + out_w*out_h*k;
	    output[out_index] = input[in_index];
         }
      }
   }
   return output;
}

/*input [0, dh2 - dh1]    copy into ==> output(w*h)   [dh1, dh2]
  	[0, dw2 - dw1]			              [dw1, dw2]*/
void stitch_feature_maps(float* input, float* output, uint32_t w, uint32_t h, uint32_t c, uint32_t dw1, uint32_t dw2, uint32_t dh1, uint32_t dh2){
   uint32_t in_w = dw2 - dw1 + 1;
   uint32_t in_h = dh2 - dh1 + 1;
   uint32_t i,j,k;
   uint32_t in_index;
   uint32_t out_index;
   for(k = 0; k < c; ++k){
      for(j = 0; j < in_h; ++j){
         for(i = 0; i < in_w; ++i){
            in_index  = i + in_w*(j + in_h*k);
            out_index  = (i + dw1) + w*(j + dh1) + w*h*k;
	    output[out_index] = input[in_index];
         }
      }
   }
}

float* get_model_input(cnn_model* model){
   return model->net->input;
}

void set_model_input(cnn_model* model, float* input){
   model->net->input = input;
}

float* get_model_output(cnn_model* model, uint32_t layer){
   return model->net->layers[layer].output;
}

uint32_t get_model_byte_size(cnn_model* model, uint32_t layer){
   return model->net->layers[layer].outputs* sizeof(float);
}


void load_image_by_number(image* img, uint32_t id){
   int32_t h = img->h;
   int32_t w = img->w;
   char filename[256];
   sprintf(filename, "data/input/%d.jpg", id);
   image im = load_image_color(filename, 0, 0);
   image sized = letterbox_image(im, w, h);
   free_image(im);
   img->data = sized.data;
}

image load_image_as_model_input(cnn_model* model, uint32_t id){
   image sized;
   sized.w = model->net->w; 
   sized.h = model->net->h; 
   sized.c = model->net->c;
   load_image_by_number(&sized, id);
   model->net->input = sized.data;
   return sized;
}

void free_image_holder(cnn_model* model, image sized){
   free_image(sized);   
}


void forward_all(cnn_model* model, uint32_t from){
   network net = *(model->net);
   int32_t i;
   for(i = from; i < net.n; ++i){
      net.index = i;
      if(net.layers[i].delta){	       
         fill_cpu(net.layers[i].outputs * net.layers[i].batch, 0, net.layers[i].delta, 1);
      }
      net.layers[i].forward(net.layers[i], net);
      net.input = net.layers[i].output;
      if(net.layers[i].truth) {
         net.truth = net.layers[i].output;
      }
   }
}

void forward_until(cnn_model* model, uint32_t from, uint32_t to){
   network net = *(model->net);
   int32_t i;
   for(i = from; i < to; ++i){
      net.index = i;
      if(net.layers[i].delta){	       
         fill_cpu(net.layers[i].outputs * net.layers[i].batch, 0, net.layers[i].delta, 1);
      }
      net.layers[i].forward(net.layers[i], net);
      net.input = net.layers[i].output;
      if(net.layers[i].truth) {
         net.truth = net.layers[i].output;
      }
   }
}


tile_region relative_offsets(tile_region large, tile_region small){
    tile_region output; 
    output.w1 = small.w1 - large.w1 ; 
    output.w2 = small.w1 - large.w1 + (small.w2 - small.w1);
    output.h1 = small.h1 - large.h1; 
    output.h2 = small.h1 - large.h1 + (small.h2 - small.h1);
    output.w = output.w2 - output.w1 + 1;
    output.h = output.h2 - output.h1 + 1;
    return output;
}
#if DATA_REUSE
void record_overlapped_output(cnn_model* model, uint32_t task_id,  uint32_t l, float* layer_output){
   ftp_parameters_reuse* ftp_para_reuse = model->ftp_para_reuse;
   network_parameters* net_para = model->net_para;
   overlapped_tile_data regions_and_data = ftp_para_reuse->output_reuse_regions[task_id][l];
   uint32_t position;
   float* data;
   tile_region offset_index;
   for(position = 0; position < 4; position++){
      tile_region overlap_index = get_region(&regions_and_data, position);
      if((overlap_index.w > 0)&&(overlap_index.h > 0)){
         if(get_size(&regions_and_data, position)>0) {
            free(get_data(&regions_and_data, position));
            set_size(&regions_and_data, position, 0);
         }
         offset_index = relative_offsets(ftp_para_reuse->output_tiles[task_id][l], overlap_index);
         data = crop_feature_maps(layer_output, 
				  ftp_para_reuse->output_tiles[task_id][l].w, 
                                  ftp_para_reuse->output_tiles[task_id][l].h, 
                                  net_para->output_maps[l].c, 
                                  offset_index.w1, offset_index.w2, 
                                  offset_index.h1, offset_index.h2);
         set_data(&regions_and_data, position, data);
         set_size(&regions_and_data, position, sizeof(float) * offset_index.w * offset_index.h * net_para->output_maps[l].c);
      }
   }
   ftp_para_reuse->output_reuse_regions[task_id][l] = regions_and_data;
}

float* stitch_reuse_output(cnn_model* model, uint32_t task_id,  uint32_t l, float* layer_output){
   ftp_parameters_reuse* ftp_para_reuse = model->ftp_para_reuse;
   network_parameters* net_para = model->net_para;
   float* stitched_data = (float*)malloc(ftp_para_reuse->input_tiles[task_id][l+1].w*ftp_para_reuse->input_tiles[task_id][l+1].h*net_para->output_maps[l].c*sizeof(float));
   tile_region offset_index;
   /*stich the central region*/
   offset_index = relative_offsets(ftp_para_reuse->input_tiles[task_id][l+1], ftp_para_reuse->output_tiles[task_id][l]);
#if DEBUG_INFERENCE
   printf("Main indexed\n");
   print_tile_region(offset_index);
#endif
   stitch_feature_maps(layer_output, stitched_data,
			ftp_para_reuse->input_tiles[task_id][l+1].w, 
                        ftp_para_reuse->input_tiles[task_id][l+1].h, 
                        net_para->output_maps[l].c, 
                        offset_index.w1, offset_index.w2, 
                        offset_index.h1, offset_index.h2);

   if(net_para->type[l+1] == POOLING_LAYER) return stitched_data;

   /*If the next layer is not pooling layer, then we need copy reuse data from adjacent partitions*/
   overlapped_tile_data regions_and_data; 
   uint32_t position;
   int32_t adjacent_id[4];
   tile_region overlap_index;
   uint32_t i = task_id/(ftp_para_reuse->partitions_w);
   uint32_t j = task_id%(ftp_para_reuse->partitions_w);

   for(position = 0; position < 4; position++){
      adjacent_id[position] = -1;
   }
   /*get the up overlapped data from tile below*/
   if((i+1)<(ftp_para_reuse->partitions_h)) adjacent_id[0] = ftp_para_reuse->task_id[i+1][j];
   /*get the left overlapped data from tile on the right*/
   if((j+1)<(ftp_para_reuse->partitions_w)) adjacent_id[1] = ftp_para_reuse->task_id[i][j+1];
   /*get the bottom overlapped data from tile above*/
   if(i>0) adjacent_id[2] = ftp_para_reuse->task_id[i-1][j];
   /*get the right overlapped data from tile on the left*/
   if(j>0) adjacent_id[3] = ftp_para_reuse->task_id[i][j-1];
   for(position = 0; position < 4; position++){
      if(adjacent_id[position]==-1) continue;
      uint32_t mirror_position = (position + 2)%4;
      regions_and_data = ftp_para_reuse->output_reuse_regions[adjacent_id[position]][l];
      overlap_index = get_region(&regions_and_data, mirror_position);
      if((overlap_index.w>0)&&(overlap_index.h>0)){
         offset_index = relative_offsets(ftp_para_reuse->input_tiles[task_id][l+1], overlap_index);
#if DEBUG_INFERENCE
         printf("side indexed\n");
         print_tile_region(offset_index);
         print_tile_region(ftp_para_reuse->input_tiles[task_id][l+1]);
         printf("side indexed, data size is %d\n", get_size(&regions_and_data, mirror_position));
#endif
         stitch_feature_maps(get_data(&regions_and_data, mirror_position), 
                        stitched_data,
			ftp_para_reuse->input_tiles[task_id][l+1].w, 
                        ftp_para_reuse->input_tiles[task_id][l+1].h, 
                        net_para->output_maps[l].c, 
                        offset_index.w1, offset_index.w2, 
                        offset_index.h1, offset_index.h2);

      }else{continue;}
   }
   return stitched_data;
}
#endif

void forward_partition(cnn_model* model, uint32_t task_id, bool is_reuse){
   network net = *(model->net);
   ftp_parameters* ftp_para = model->ftp_para;
   /*network_parameters* net_para = model->net_para;*/
   uint32_t l;
   for(l = 0; l < ftp_para->fused_layers; l++){
      net.layers[l].h = ftp_para->input_tiles[task_id][l].h;
      net.layers[l].out_h = (net.layers[l].h/net.layers[l].stride); 
      net.layers[l].w = ftp_para->input_tiles[task_id][l].w;
      net.layers[l].out_w = (net.layers[l].w/net.layers[l].stride);
      net.layers[l].outputs = net.layers[l].out_h * net.layers[l].out_w * net.layers[l].out_c; 
      net.layers[l].inputs = net.layers[l].h * net.layers[l].w * net.layers[l].c; 
   }
   uint32_t to_free = 0;
   float * cropped_output;
#if DATA_REUSE
   uint32_t to_free_next_layer_input = 0;
   float* stitched_next_layer_input = NULL; 
   ftp_parameters_reuse* ftp_para_reuse = model->ftp_para_reuse;
   if((model->ftp_para_reuse->schedule[task_id] == 1)&&is_reuse){
      for(l = 0; l < ftp_para_reuse->fused_layers; l++){
         net.layers[l].h = ftp_para_reuse->input_tiles[task_id][l].h;
         net.layers[l].out_h = (net.layers[l].h/net.layers[l].stride); 
         net.layers[l].w = ftp_para_reuse->input_tiles[task_id][l].w;
         net.layers[l].out_w = (net.layers[l].w/net.layers[l].stride);
         net.layers[l].outputs = net.layers[l].out_h * net.layers[l].out_w * net.layers[l].out_c; 
         net.layers[l].inputs = net.layers[l].h * net.layers[l].w * net.layers[l].c; 
      }
   }
#endif

   for(l = 0; l < ftp_para->fused_layers; l++){
      net.layers[l].forward(net.layers[l], net);
      if (to_free == 1) {
         free(cropped_output); 
         to_free = 0; 
         /*Free the memory allocated by the crop_feature_maps function call;*/
      }
      /*
        The effective region is actually shrinking after each convolutional layer 
        because of padding effects.
        So for the calculation of next layer, the boundary pixels should be removed.
      */
      tile_region tmp;
      if(net.layers[l].type == CONVOLUTIONAL){
         tmp = relative_offsets(ftp_para->input_tiles[task_id][l], 
                                       ftp_para->output_tiles[task_id][l]);
#if DATA_REUSE
         if((model->ftp_para_reuse->schedule[task_id] == 1)&&is_reuse){
            tmp = relative_offsets(ftp_para_reuse->input_tiles[task_id][l], 
                                       ftp_para_reuse->output_tiles[task_id][l]);
         }
#endif
         cropped_output = crop_feature_maps(net.layers[l].output, 
                      net.layers[l].out_w, net.layers[l].out_h, net.layers[l].out_c,
                      tmp.w1, tmp.w2, tmp.h1, tmp.h2);
         to_free = 1;
      } else {cropped_output = net.layers[l].output;}  
      net.input = cropped_output;
#if DATA_REUSE
      record_overlapped_output(model, task_id, l, cropped_output);/*Record the generated overlapped regions for reuse*/
      if((model->ftp_para_reuse->schedule[task_id] == 1)&&is_reuse){
         if (to_free_next_layer_input == 1) {
            free(stitched_next_layer_input);
            to_free_next_layer_input = 0;
         }
         if(l < (ftp_para_reuse->fused_layers - 1)){
            stitched_next_layer_input = stitch_reuse_output(model, task_id, l, cropped_output);
            to_free_next_layer_input = 1;
         }
         net.input = stitched_next_layer_input;
      }
#endif
   }
   if (to_free == 1) free(cropped_output);

#if DATA_REUSE
   if (to_free_next_layer_input == 1) free(stitched_next_layer_input);
#endif
}

void draw_object_boxes(cnn_model* model, uint32_t id){
   network net = *(model->net);
   image sized;
   sized.w = net.w; sized.h = net.h; sized.c = net.c;
   load_image_by_number(&sized, id);
   image **alphabet = load_alphabet();
   list *options = read_data_cfg((char*)"data/coco.data");
   char *name_list = option_find_str(options, (char*)"names", (char*)"data/names.list");
   char **names = get_labels(name_list);
   char filename[256];
   char outfile[256];
   float thresh = .24;
   float hier_thresh = .5;
   float nms=.3;
   sprintf(filename, "data/input/%d.jpg", id);
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


