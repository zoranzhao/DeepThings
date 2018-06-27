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
   network* net;/*network is from Darknet*/
} cnn_model;

#define CONV_LAYER CONVOLUTIONAL /*CONVOLUTIONAL is from Darknet*/
#define POOLING_LAYER MAXPOOL /*MAXPOOL is from Darknet*/

cnn_model* load_cnn_model(char* cfg, char* weights);
void forward_partition(cnn_model* model, uint32_t task_id);
void forward_all(cnn_model* model, uint32_t from);

#endif
