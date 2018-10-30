/*
 * main.c
 *
 *  Created on: Oct 30, 2018
 *      Author: eashvla
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cublas.h>
#include <cublas_api.h>
#include <cublas_v2.h>
#include <sys/time.h>
#include <cudnn.h>
#include "DeepLearning.h"
#include "utils.h"
#include <syslog.h>
#define FP "./layers.info"

setlogmask (LOG_UPTO (LOG_DEBUG));
openlog ("DeepLearning", LOG_CONS | LOG_PID | LOG_NDELAY, LOG_LOCAL1);

int main(int argc, char **argv) {
	FILE* fp;
	int num_layers;
	struct layer* layers = NULL;
	struct descriptor* desc = NULL;
	LayerType type;
	cudnnHandle_t cudnn;
	cublasHandle_t cublas;
	cudnnStatus_t status;
	int batch_size = 32;
	float* input_image;
	float* yhat;
	float* one_vector;
	float cost;

	if ((fp=fopen(FP, "r"))==NULL) {
		syslog(LOG_ERROR, "Unable to find the layer information file. Terminating the program");
		exit(1);
	}

	fscanf(fp, "%d \n", &num_layers);
	if (num_layers < 0 || num_layers > 10) {
		syslog(LOG_ERROR, "Number of layers must be between 1 and 10. Obtained %d. Terminating", num_layers);
		exit(1);
	}
	(struct layer*) layers = (struct layer*)calloc(sizeof(struct layer)*num_layers);
	if (layers == NULL) {
		syslog(LOG_ERROR, "Unable to allocate memory for layering info collection");
		exit(1);
	}
	for (int i=0; i < num_layers; i++) {
		fscanf(fp, "%d ", &layers[i].type);
		if(layers[i].type = FULLYCONNECTED) {
			struct fcLayer* fc = &layers[i].fc_layer;
			fscanf(fp, "%d %d %d\n", fc->input_size, fc->size, fc->activation);
			get_matrix(&fc->weights, fc->input_size, fc->size,1);
		} else {
			struct convLayer* cl = &layers[i].conv_layer;
			fscanf(fp, "%d %d %d %d %d %d %d\n", cl->filter_size, cl->padding, cl->stride,
												 cl->maxpool, cl->num_channels, cl->algorithm, cl->mode);
			get_matrix(&cl->filter, cl->filter_size, cl->filter_size, 1);
		}
	}

	if ((status = cudnnCreate(&cudnn)) != CUDNN_STATUS_SUCCESS) {
		syslog(LOG_ERROR, "Unable to create CUDA handlers. Terminating the program");
		exit(1);
	}

	if ((status = cublasCreate(&cublas))) {
		syslog(LOG_ERROR, "Unable to create CUDA handlers. Terminating the program");
		exit(1);
	}

	get_matrix(&input_image, IMAGE_HEIGHT*IMAGE_WIDTH, batch_size, 1);
	status = setup_descriptors (&desc, num_layers, layers);
	if(status != 0) {
		syslog(LOG_ERROR, "Error while Descriptor Setup. Terminating the program");
		exit(1);
	}
	configure_descriptors(&cudnn, desc, num_layers, layers, batch_size);
	if(status != 0) {
			syslog(LOG_ERROR, "Error while Descriptor Config. Terminating the program");
			exit(1);
	}

	int t, n, c, h, w;
	status = cudnnGetTensor4dDescriptor(*(desc[num_layers-1].output_desc), &t, &n, &c, &h, &w, NULL, NULL, NULL, NULL);
	if(status != 0) {
		syslog(LOG_ERROR, "Error while determining Output vec size. Terminating the program");
		exit(1);
	}
	get_matrix(&yhat, n*c, h*w, 1);
	float* one_vector = (float*) calloc(sizeof(float)* n*c*h*w);
	for (int i=0; i< n*c*h*w; i++)
		one_vector[i] = 1;

	allocate_memory(desc, layers, num_layers, batch_size) ;
	if(status != 0) {
		syslog(LOG_ERROR, "Error while allocating Memory. Terminating the program");
		exit(1);
	}
	copy_input_to_device(desc, layers, num_layers, input_image, batch_size);
	if(status != 0) {
		syslog(LOG_ERROR, "Error while Copying data to Device. Terminating the program");
		exit(1);
	}

	status = feedforward(&cudnn, &cublas ,  desc, layers, num_layers, batch_size);
	if(status != 0) {
		syslog(LOG_ERROR, "Error in Feed-forward. Terminating the program");
		exit(1);
	}

	status = computecost(desc[num_layers-1].d_output, yhat,  one_vector, n*c*h*w, cublas, &cost);
	if(status != 0) {
		syslog(LOG_ERROR, "Error in Compute cost. Terminating the program");
		exit(1);
	}

	status = destroy_descriptors (desc, num_layers);
	if(status != 0) {
		syslog(LOG_ERROR, "Error in Cleanup. Terminating the program");
		exit(1);
	}

	printf("The cost of the Optimization Function is %2.3f \n", cost);
}
