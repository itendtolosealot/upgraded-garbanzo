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
#include <sys/syslog.h>
#include "DeepLearning.h"

__global__ void cross_entropy(int array_size, float* y, float* yhat)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  float res;
  if (i < array_size) {
	  if((yhat[i] == 0)|| (y[i]==0) || (y[i]==1) || (yhat[i] == 1))
		  res = 0;
	  else if (y[i]  > 1)
		  res = __logf(1-yhat[i])*y[i] + __logf(yhat[i])*(1-y[i]);
	  y[i] = res;
  }
}


int setup_descriptors ( struct descriptor** desc, int num_layers, struct layer *layers) {
	struct descriptor* d;
	cudnnStatus_t status;

	d = (struct descriptor*) malloc(sizeof(struct descriptor)*num_layers);
	if (d == NULL)
		return 1000;
	for(int i=0;i< num_layers;i++) {
		if(layers[i].type==CONVOLUTION) {
			d[i].valid = true;
			status = cudnnCreateTensorDescriptor(d[i].input_desc);
			if(status != CUDNN_STATUS_SUCCESS) return (int)status;

			status = cudnnCreateTensorDescriptor(d[i].output_desc);
			if(status != CUDNN_STATUS_SUCCESS) return (int)status;

			status = cudnnCreateFilterDescriptor(d[i].filter_desc);
			if(status != CUDNN_STATUS_SUCCESS) return (int)status;

			status = cudnnCreateConvolutionDescriptor(d[i].conv_desc);
			if(status != CUDNN_STATUS_SUCCESS) return (int)status;

			d[i].d_weights = NULL;

		} else {
			d[i].valid = false;
			d[i].input_desc= NULL;
			d[i].filter_desc=NULL;
			d[i].output_desc=NULL;
			d[i].conv_desc=NULL;
			status = cudnnCreateActivationDescriptor(d[i].acti_desc);
		}
	}
	*(desc) = d;
	return 0;
}

int destroy_descriptors (struct descriptor* desc, int num_layers) {
	cudnnStatus_t status;
	for(int i=0;i< num_layers;i++) {
		if(desc[i].valid) {
			status = cudnnDestroyTensorDescriptor(*(desc[i].input_desc));
			if(status != CUDNN_STATUS_SUCCESS) return (int)status;
			status = cudnnDestroyTensorDescriptor(*(desc[i].output_desc));
			if(status!= CUDNN_STATUS_SUCCESS) return (int)status;
			status = cudnnDestroyFilterDescriptor(*(desc[i].filter_desc));
			if(status!= CUDNN_STATUS_SUCCESS) return (int)status;
			status = cudnnDestroyConvolutionDescriptor(*(desc[i].conv_desc));
			if(status!= CUDNN_STATUS_SUCCESS) return (int)status;
		} else {
			cudaFree(desc[i].d_weights);
		}
	}
	free(desc);
	return 0;
}

int configure_descriptors(cudnnHandle_t* handle, struct descriptor* desc, int num_layers, struct layer *layers, int batch_size) {
	cudnnStatus_t status;
	int n,c,h,w;
	int output_img_width,output_img_height;
	for (int i=0; i < num_layers;i++) {
		if (desc[i].valid) {
			if(i==0) {
				status = cudnnSetTensor4dDescriptor(*(desc[i].input_desc), CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, 1, IMAGE_HEIGHT, IMAGE_WIDTH);
			} else {
				cudnnDataType_t t;
				status = cudnnGetTensor4dDescriptor(*(desc[i-1].output_desc), &t, &n, &c, &h, &w, NULL, NULL, NULL, NULL);
				status = cudnnSetTensor4dDescriptor(*(desc[i].input_desc), CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n,c,h,w);
			}
			int nc = layers[i].conv_layer.num_channels;
			int size = layers[i].conv_layer.filter_size;
			int pad= layers[i].conv_layer.padding;
			int stride = layers[i].conv_layer.stride;
			int input_img_width = (i==0) ? IMAGE_WIDTH : w;
			int input_img_height = (i==0) ? IMAGE_HEIGHT:h;
			status = cudnnSetFilter4dDescriptor(*(desc[i].filter_desc), CUDNN_DATA_FLOAT,CUDNN_TENSOR_NCHW, 1, nc,size,size);
			if(status != CUDNN_STATUS_SUCCESS) return (int)status;
			status = cudnnSetConvolution2dDescriptor(*(desc[i].conv_desc), pad, pad, stride, stride, 1,1, CUDNN_CROSS_CORRELATION,CUDNN_DATA_FLOAT);
			if (status != CUDNN_STATUS_SUCCESS) return (int)status;
			status = cudnnGetConvolution2dForwardOutputDim(*(desc[i].conv_desc), *(desc[i].input_desc), *(desc[i].filter_desc), &n, &c, &h, &w);
			if (status != CUDNN_STATUS_SUCCESS) return (int)status;
			status = cudnnSetTensor4dDescriptor(*(desc[i].output_desc), CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n,c,h,w);
			if (status != CUDNN_STATUS_SUCCESS) return (int)status;
			status = cudnnGetConvolutionForwardAlgorithm(*handle, *(desc[i].input_desc), *(desc[i].filter_desc),
														*(desc[i].conv_desc), *(desc[i].output_desc),
														CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,0,
														&desc[i].algo_desc);
			if(status != CUDNN_STATUS_SUCCESS) return (int)status;
			status = cudnnGetConvolutionForwardWorkspaceSize(*handle, *(desc[i].input_desc),
															*(desc[i].filter_desc),*(desc[i].conv_desc),
															*(desc[i].output_desc), desc[i].algo_desc,
															&desc[i].workspace_size);
			if(status != CUDNN_STATUS_SUCCESS) return (int)status;
		} else {
			status = cudnnSetActivationDescriptor(*(desc[i].acti_desc), layers[i].fc_layer.activation,
													CUDNN_NOT_PROPAGATE_NAN, 0.5);

			if(status != CUDNN_STATUS_SUCCESS) return (int)status;
		}
	}
	return 0;
}

int allocate_memory(struct descriptor* desc, struct layer* layers, int num_layers, int batch_size) {
	int n,c,h,w;
	cudnnStatus_t status;
	cudaError_t stat;
	cudnnDataType_t t;
	cudnnTensorFormat_t format;
	for (int i=0;i<num_layers;i++) {
		if(desc[i].valid) {
			if(i==0) {
				cudaMalloc(&desc[i].d_input, batch_size*IMAGE_HEIGHT*IMAGE_WIDTH*sizeof(float));
			} else {
				if (desc[i - 1].valid) {
					status = cudnnGetTensor4dDescriptor(*(desc[i - 1].output_desc), &t, &n, &c, &h, &w,
						NULL, NULL, NULL, NULL);
				}
				else {
					n = batch_size;
					c = 1;
					h = 1;
					w = layers[i - 1].fc_layer.size;
				}
				if(status != CUDNN_STATUS_SUCCESS) return (int)status;
				stat = cudaMalloc(&desc[i].d_input, n*c*h*w*sizeof(float));
				if(stat != cudaSuccess) return stat;
			}
			status = cudnnGetFilter4dDescriptor(*(desc[i].filter_desc), &t, &format, &n,&c,&h,&w);
			cudaMalloc(&desc[i].d_filter, n*c*h*w*sizeof(float));
			if(i==num_layers-1) {
				status = cudnnGetTensor4dDescriptor(*(desc[i].output_desc), &t, &n, &c, &h, &w,
													NULL, NULL, NULL, NULL);
				if(status != CUDNN_STATUS_SUCCESS) return (int)status;
				stat = cudaMalloc(&desc[i].d_output,n*c*h*w*sizeof(float));
				if(stat != cudaSuccess) return stat;
			}
			stat = cudaMalloc(&desc[i].d_workspace,desc[i].workspace_size);
			if(stat != cudaSuccess) return stat;

		} else {
				stat = cudaMalloc(&desc[i].d_input, layers[i].fc_layer.input_size*sizeof(float));
				if(stat != cudaSuccess) return stat;
				stat = cudaMalloc(&desc[i].d_weights,
							(layers[i].fc_layer.input_size)/batch_size*layers[i].fc_layer.size*sizeof(float));
				if(stat != cudaSuccess) return stat;
				if(i==num_layers-1) {
					stat = cudaMalloc(&desc[i].d_output, batch_size*layers[i].fc_layer.size*sizeof(float));
					if(stat != cudaSuccess) return stat;
				}
		}


	}
	return 0;
}

int copy_input_to_device(struct descriptor* desc, struct layer* layers, int num_layers, float* input_image, int batch_size)
{
	cudnnStatus_t status;
	cudaError_t stat;
	cudnnDataType_t t;
	cudnnTensorFormat_t format;
	int n,c,h,w;

	stat = cudaMemcpy(desc[0].d_input, input_image, sizeof(float)*batch_size*IMAGE_WIDTH*IMAGE_HEIGHT, cudaMemcpyHostToDevice);
	for(int i=0; i< num_layers; i++) {
		if(desc[i].valid)  {
			status = cudnnGetFilter4dDescriptor(*(desc[i].filter_desc), &t, &format, &n,&c,&h,&w);
			if(status != CUDNN_STATUS_SUCCESS) return stat;
			stat = cudaMemcpy(desc[i].d_filter, layers[i].conv_layer.filter,
								sizeof(float)*n*c*h*w, cudaMemcpyHostToDevice);
			if(stat != cudaSuccess) return stat;
		} else {
			stat = cudaMemcpy(desc[i].d_weights, layers[i].fc_layer.weights ,
						sizeof(float)*layers[i].fc_layer.input_size*layers[i].fc_layer.size,
						cudaMemcpyHostToDevice);
			if(stat != cudaSuccess) return stat;
		}
	}
	return 0;

}


int feedforward(cudnnHandle_t* cudnn, 	cublasHandle_t* handle, struct descriptor* desc, struct layer *layers, int num_layers, int batch_size)
{
	cudnnStatus_t status;
	cublasStatus_t stat;
	float* output_array;
	const float alpha=1, beta=0;
	for(int i=0;i < num_layers;i++) {
        output_array = (i < num_layers-1) ? desc[i+1].d_input:desc[i].d_output;
		if(desc[i].valid) {
				status = cudnnConvolutionForward(*cudnn,&alpha, *(desc[i].input_desc), desc[i].d_input,
											*(desc[i].filter_desc),desc[i].d_filter, *(desc[i].conv_desc),
											 desc[i].algo_desc, desc[i].d_workspace,desc[i].workspace_size,
											 &beta, *(desc[i].output_desc), output_array);
				if(status != CUDNN_STATUS_SUCCESS) return (int)status;
		} else {
				stat =  cublasSgemm(*handle,
									CUBLAS_OP_N,
									CUBLAS_OP_N,
									layers[i].fc_layer.size,
									batch_size,
									(int) ((layers[i].fc_layer.input_size)/batch_size),
									&alpha,
									desc[i].d_weights,
									layers[i].fc_layer.size,
									desc[i].d_input,
									(int)((layers[i].fc_layer.input_size)/batch_size ),
									&beta,
									output_array,
									layers[i].fc_layer.size);
				if (stat != CUBLAS_STATUS_SUCCESS) return (int) stat;

				status = cudnnActivationForward(*cudnn,
												*(desc[i].acti_desc),
												&alpha,
												*(desc[i].output_desc),
												(void *)output_array,
												&beta,
												*(desc[i].output_desc),
												(void *) output_array);
				if(status != CUDNN_STATUS_SUCCESS) return (int)status;
			}
		}
	return 0;
}


int computecost(float* y, float* yhat, float* ones_vector, int size, cublasHandle_t* handle, float* cost) {
	cudaError_t status;
	cublasStatus_t stat;
	cross_entropy<<<(size+255)/256, 256>>>(size, y, yhat);
    status = cudaDeviceSynchronize();
    if (status != cudaSuccess) return (int) status;
    stat = cublasSdot_v2(*handle, size, ones_vector,1, y, 1, cost);
	if (stat != CUBLAS_STATUS_SUCCESS) return (int) stat;
	return 0;
}