/*
 * DeepLearning.h
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
#include <unistd.h>

#include <cublas_v2.h>
#include <sys/time.h>
#include <cudnn.h>
#ifndef DEEPLEARNING_H_
#define DEEPLEARNING_H_
#define IMAGE_HEIGHT 16
#define IMAGE_WIDTH 16

#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

enum LayerType {CONVOLUTION, FULLYCONNECTED};
enum FailureType {NONE, CUDNN, CUBLAS};
struct convLayer {
	int filter_size;
	int padding;
	int stride;
	int maxpool;
	int num_channels;
	float* filter;
	cudnnConvolutionFwdAlgo_t algorithm;
	cudnnConvolutionMode_t mode;
};

struct fcLayer {
	int size;
	int input_size;
	float* weights;
	cudnnActivationMode_t activation;
};

struct Status {
	FailureType failure;
	int layer;
	cudnnStatus_t  cudnn_stat;
	cublasStatus_t cublas_stat;
};

struct layer {
	LayerType type;
	union {
		struct fcLayer fc_layer;
		struct convLayer conv_layer;
	};
};

struct descriptor {
	bool valid;
	cudnnTensorDescriptor_t input_desc;
	cudnnTensorDescriptor_t y_desc;
	cudnnTensorDescriptor_t output_desc;
	cudnnFilterDescriptor_t filter_desc;
	cudnnConvolutionDescriptor_t conv_desc;
	cudnnConvolutionFwdAlgo_t algo_desc;
	cudnnActivationDescriptor_t acti_desc;
	unsigned long workspace_size;
	float* d_input;
	float* d_filter;
	float* d_y;
	float* d_output;
	float* d_weights;
	float* d_workspace;
};
int setup_descriptors ( struct descriptor** desc, int num_layers, struct layer *layers);
int destroy_descriptors (struct descriptor* desc, int num_layers);
int configure_descriptors(cudnnHandle_t* handle, struct descriptor* desc, int num_layers, struct layer *layers, int batch_size);
int allocate_memory(struct descriptor* desc, struct layer* layers, int num_layers, int batch_size) ;
int copy_input_to_device(struct descriptor* desc, struct layer* layers, int num_layers, float* input_image, int batch_size);
struct Status feedforward(cudnnHandle_t* cudnn, 	cublasHandle_t* handle, struct descriptor* desc, struct layer *layers, int num_layers, int batch_size);
int computecost(float* y, float* yhat, float* ones_vector, int size, cublasHandle_t handle, float* cost);







#endif /* DEEPLEARNING_H_ */
