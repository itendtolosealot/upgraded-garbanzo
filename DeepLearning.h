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
#include "mkl.h"
#include <cublas_v2.h>
#include <time.h>
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
enum FailureType {NONE, CUDNN, CUBLAS, CUDA};
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
	float* bias;
	cudnnActivationMode_t activation;
};

struct Status {
	FailureType failure;
	int layer;
	cudnnStatus_t  cudnn_stat;
	cublasStatus_t cublas_stat;
	cudaError_t cuda_stat;
};

struct layer {
	LayerType type;
	union {
		struct fcLayer fc_layer;
		struct convLayer conv_layer;
	};
};
/* cost descriptor variables would be used in cost computation */
struct cost_descriptor {
	float* d_out;	// d_out represents the output of the NN 
	float* d_dout;	// d_dout represents the gradient
	float* d_yhat;	// d_yhat represents the exponentiated output, i.e., FOR EACH EXAMPLE, d_yhat[i] = exp(d_out[i])/(\sum_{j=0}^{output_size} exp(d_out[j]))
	float* d_y;		// d_y represents the labelled output. It indicates the true output for the example.
	float* d_one_vec;
	float* d_sum_exp;
	float* h_one_vec; // Vector on the host that would be copied to the device
	float* h_y;		// Vector on the host that would be copied to the device
};

struct descriptor {
	bool valid;
	cudnnTensorDescriptor_t input_desc;
	cudnnTensorDescriptor_t din_desc;
	cudnnTensorDescriptor_t y_desc;
	cudnnTensorDescriptor_t dy_desc;
	cudnnTensorDescriptor_t dout_desc;
	cudnnFilterDescriptor_t filter_desc;
	cudnnFilterDescriptor_t dfilter_desc;
	cudnnConvolutionDescriptor_t conv_desc;
	cudnnConvolutionFwdAlgo_t algo_desc;
	cudnnActivationDescriptor_t acti_desc;
	cudnnTensorDescriptor_t output_desc;
	unsigned long workspace_size;
	// Naming convention: The first 'd' stands for "device" indiacting that the array lives in the device. The second d, whenever used 
	// stands for delta or gradient of the cost with respect to that variable. 
	float* d_input;
	float* d_filter;
	float* d_y;
	float* d_weights;
	float* d_bias;
	float* d_workspace;
	float* d_din;
	float* d_df;
	float* d_dy;
	float* d_dw;
	float* d_db;
};
int setup_descriptors ( struct descriptor** desc, int num_layers, struct layer *layers);
int destroy_descriptors (struct descriptor* desc, struct cost_descriptor cost, int num_layers);
int configure_descriptors(cudnnHandle_t* handle, struct descriptor* desc, int num_layers, struct layer *layers, int batch_size);
int allocate_memory(struct descriptor* desc, struct cost_descriptor cost, struct layer* layers, int num_layers, int batch_size) ;
int copy_input_to_device(struct descriptor* desc, struct cost_descriptor cost, struct layer* layers, int num_layers, float* input_image, int batch_size);
struct Status feedforward(cudnnHandle_t* cudnn, cublasHandle_t* handle, struct descriptor* desc, struct cost_descriptor cost, struct layer *layers, int num_layers, int batch_size);
int computecost(struct cost_descriptor cost, int batch_size, int output_size, cublasHandle_t handle, float* total_cost);







#endif /* DEEPLEARNING_H_ */
