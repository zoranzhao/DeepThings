#include <stdio.h>
#include <time.h>
#include <assert.h>
#include "network_dist.h"
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

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

//	char datafile[80];

//	sprintf(datafile, "data_%d.bin", i);
//	FILE *file = fopen(datafile, "wb");
//	fwrite(l.output, sizeof(float), l.outputs, file);
//	fclose(file);


        //printf("Float is %d, Layer %s spends %f, output data size is: %d\n", sizeof(float), get_layer_string(l.type), t2 - t1, l.outputs);


void write_layer(network *netp, int idx)
{
    network net = *netp;
    layer l = net.layers[idx];
    FILE *p_file;

    char filename[50];
    sprintf(filename, "%s_%d.dat", get_layer_string(net.layers[idx + 1].type), idx + 1);
    p_file = fopen(filename, "wb");
 
    fwrite(l.output, sizeof(float), l.outputs, p_file); 
    fclose(p_file);
}

void read_layer(network *netp, int idx)
{
    network net = *netp;
    layer l = net.layers[idx];
    FILE *p_file;

    char filename[50];
    sprintf(filename, "%s_%d.dat", get_layer_string(net.layers[idx].type), idx);
    p_file = fopen(filename, "rb");
    fread(net.input, sizeof(float), l.inputs, p_file); 
    fclose(p_file);

}


inline void forward_network_dist(network *netp)
{
    network net = *netp;
    int i;
    //Network input
    //net.input
    double read_t = 0;
    double write_t = 0;

    for(i = 0; i < net.n; ++i){//Iteratively execute the layers
        net.index = i;
        //layer l = net.layers[i];
        if(net.layers[i].delta){	       
            fill_cpu(net.layers[i].outputs * net.layers[i].batch, 0, net.layers[i].delta, 1);
        }
        net.layers[i].forward(net.layers[i], net);
        net.input = net.layers[i].output;  //Layer output
        if(net.layers[i].truth) {
            net.truth = net.layers[i].output;
        }
        //printf("Index %d, Layer %s, input data size is: %d, output data size is: %d\n", i, get_layer_string(net.layers[i].type), net.layers[i].inputs, net.layers[i].outputs);
/*
	if(i > 0){
            double t1 = what_time_is_it_now();
	    read_layer(netp, i);
            double t2 = what_time_is_it_now();
	    read_t = read_t + t2 - t1; 
	}
	if(i < (net.n-1)){
            double t1 = what_time_is_it_now();
	    write_layer(netp, i);
            double t2 = what_time_is_it_now();
	    write_t = write_t + t2 - t1; 
	}
*/
    }
    //printf("Writing time is: %lf, reading time is: %lf\n", read_t, write_t);
    //calc_network_cost(netp);
}

inline float *network_predict_dist(network *net, float *input)
{
    network orig = *net;
    net->input = input;
    net->truth = 0;
    net->train = 0;
    net->delta = 0;
    forward_network_dist(net);
    float *out = net->output;
    *net = orig;
    return out;
}
