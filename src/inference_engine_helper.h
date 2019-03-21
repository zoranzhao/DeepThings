#ifndef INFERENCE_ENGINE_HELPER_H
#define INFERENCE_ENGINE_HELPER_H

/*Include header files from inference engine*/
#include "darknet.h"
#include "image.h"
#include "layer.h"
#include "data.h"
#include "tree.h"
#include "network.h"
#include "image.h"
#include "data.h"
#include "utils.h"
#include "blas.h"
#include "crop_layer.h"
#include "connected_layer.h"
#include "gru_layer.h"
#include "rnn_layer.h"
#include "crnn_layer.h"
#include "local_layer.h"
#include "convolutional_layer.h"
#include "activation_layer.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "normalization_layer.h"
#include "batchnorm_layer.h"
#include "maxpool_layer.h"
#include "reorg_layer.h"
#include "avgpool_layer.h"
#include "cost_layer.h"
#include "softmax_layer.h"
#include "dropout_layer.h"
#include "route_layer.h"
#include "shortcut_layer.h"
#include "parser.h"
#include "data.h"
#include "option_list.h"
/*Include header files from inference engine*/

#include "configure.h"
#include "ftp.h"

typedef struct cnn_model_wrapper{
   ftp_parameters* ftp_para;
   network_parameters* net_para;
#if DATA_REUSE
   ftp_parameters_reuse* ftp_para_reuse;
#endif
   network* net;/*network is from Darknet*/
} cnn_model;

#define CONV_LAYER CONVOLUTIONAL /*CONVOLUTIONAL is from Darknet*/
#define POOLING_LAYER MAXPOOL /*MAXPOOL is from Darknet*/
#define image_holder image


cnn_model* load_cnn_model(char* cfg, char* weights);
void forward_partition(cnn_model* model, uint32_t task_id, bool is_reuse);
image_holder load_image_as_model_input(cnn_model* model, uint32_t id);
void free_image_holder(cnn_model* model, image_holder sized);
void forward_all(cnn_model* model, uint32_t from);
void forward_until(cnn_model* model, uint32_t from, uint32_t to);
void draw_object_boxes(cnn_model* model, uint32_t id);

float* crop_feature_maps(float* input, uint32_t w, uint32_t h, uint32_t c, uint32_t dw1, uint32_t dw2, uint32_t dh1, uint32_t dh2);
void stitch_feature_maps(float* input, float* output, uint32_t w, uint32_t h, uint32_t c, uint32_t dw1, uint32_t dw2, uint32_t dh1, uint32_t dh2);

float* get_model_input(cnn_model* model);
void set_model_input(cnn_model* model, float* input);
float* get_model_output(cnn_model* model, uint32_t layer);
uint32_t get_model_byte_size(cnn_model* model, uint32_t layer);
tile_region relative_offsets(tile_region large, tile_region small);

#endif
