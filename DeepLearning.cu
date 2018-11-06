#include <stdio.h>
#include <assert.h>
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
#include "utils.h"

__global__ void cross_entropy(int array_size, float* y, float* yhat, float* result)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  float res = 0;
  if (i < array_size) {

	  if((yhat[i] == 0)|| (yhat[i] == 1))
		  res = 0;
	  else
		  res = __logf(1-yhat[i])*y[i] + __logf(yhat[i])*(1-y[i]);
	  result[i] = res;
	//  result = y[i]+ yhat[i];
  }
}


int setup_descriptors ( struct descriptor** desc, int num_layers, struct layer *layers) {
	struct descriptor* d;
	cudnnStatus_t status;

	d = (struct descriptor*) malloc(sizeof(descriptor)*num_layers);
	if (d == NULL)
		return 1000;
	for(int i=0;i< num_layers;i++) {
		if(layers[i].type==CONVOLUTION) {
			d[i].valid = true;
			status = cudnnCreateTensorDescriptor(&d[i].input_desc);
			if(status != CUDNN_STATUS_SUCCESS) return (int)status;

			status = cudnnCreateTensorDescriptor(&d[i].output_desc);
			if(status != CUDNN_STATUS_SUCCESS) return (int)status;

			status = cudnnCreateFilterDescriptor(&d[i].filter_desc);
			if(status != CUDNN_STATUS_SUCCESS) return (int)status;

			status = cudnnCreateConvolutionDescriptor(&d[i].conv_desc);
			if(status != CUDNN_STATUS_SUCCESS) return (int)status;

			d[i].d_weights = NULL;

		} else {
			d[i].valid = false;
			status = cudnnCreateTensorDescriptor(&d[i].y_desc);
			if(status != CUDNN_STATUS_SUCCESS) return (int)status;
			status = cudnnCreateTensorDescriptor(&d[i].output_desc);
			if(status != CUDNN_STATUS_SUCCESS) return (int)status;
			status = cudnnCreateActivationDescriptor(&d[i].acti_desc);
			if(status != CUDNN_STATUS_SUCCESS) return (int)status;
		}
	}
	*(desc) = d;
	return 0;
}

int destroy_descriptors (struct descriptor* desc, int num_layers) {
	cudnnStatus_t status;
	for(int i=0;i< num_layers;i++) {
		if(desc[i].valid) {
			status = cudnnDestroyTensorDescriptor((desc[i].input_desc));
			if(status != CUDNN_STATUS_SUCCESS) return (int)status;
			status = cudnnDestroyTensorDescriptor((desc[i].output_desc));
			if(status!= CUDNN_STATUS_SUCCESS) return (int)status;
			status = cudnnDestroyFilterDescriptor((desc[i].filter_desc));
			if(status!= CUDNN_STATUS_SUCCESS) return (int)status;
			status = cudnnDestroyConvolutionDescriptor((desc[i].conv_desc));
			if(status!= CUDNN_STATUS_SUCCESS) return (int)status;

		} else {
			if (desc[i].d_weights != NULL) cudaFree(desc[i].d_weights);
			if (desc[i].d_y != NULL) cudaFree(desc[i].d_y);
			cudnnDestroyTensorDescriptor(desc[i].y_desc);
			cudnnDestroyTensorDescriptor(desc[i].output_desc);
			cudnnDestroyActivationDescriptor(desc[i].acti_desc);
		}

		if(desc[i].d_input != NULL) cudaFree(desc[i].d_input);
		if(desc[i].d_filter != NULL) cudaFree(desc[i].d_filter);
		if(desc[i].d_output != NULL) cudaFree(desc[i].d_output);
		if(desc[i].d_workspace != NULL) cudaFree(desc[i].d_workspace);
	}
	free(desc);
	return 0;
}

int configure_descriptors(cudnnHandle_t* handle, struct descriptor* desc, int num_layers, struct layer *layers, int batch_size) {
	cudnnStatus_t status;
	int n,c,h,w;
	for (int i=0; i < num_layers;i++) {
		if (desc[i].valid) {
			if(i==0) {
				status = cudnnSetTensor4dDescriptor((desc[i].input_desc), CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, 1, IMAGE_HEIGHT, IMAGE_WIDTH);
			} else {
				cudnnDataType_t t;
				status = cudnnGetTensor4dDescriptor((desc[i-1].output_desc), &t, &n, &c, &h, &w, NULL, NULL, NULL, NULL);
				status = cudnnSetTensor4dDescriptor((desc[i].input_desc), CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n,c,h,w);
			}
			int nc = layers[i].conv_layer.num_channels;
			int size = layers[i].conv_layer.filter_size;
			int pad= layers[i].conv_layer.padding;
			int stride = layers[i].conv_layer.stride;
			status = cudnnSetFilter4dDescriptor((desc[i].filter_desc), CUDNN_DATA_FLOAT,CUDNN_TENSOR_NCHW, 1, nc,size,size);
			if(status != CUDNN_STATUS_SUCCESS) return (int)status;
			status = cudnnSetConvolution2dDescriptor((desc[i].conv_desc), pad, pad, stride, stride, 1,1, CUDNN_CROSS_CORRELATION,CUDNN_DATA_FLOAT);
			if (status != CUDNN_STATUS_SUCCESS) return (int)status;
			status = cudnnGetConvolution2dForwardOutputDim((desc[i].conv_desc), (desc[i].input_desc), (desc[i].filter_desc), &n, &c, &h, &w);
			if (status != CUDNN_STATUS_SUCCESS) return (int)status;
			status = cudnnSetTensor4dDescriptor((desc[i].output_desc), CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n,c,h,w);
			if (status != CUDNN_STATUS_SUCCESS) return (int)status;
			status = cudnnGetConvolutionForwardAlgorithm(*handle, (desc[i].input_desc), (desc[i].filter_desc),
														(desc[i].conv_desc), (desc[i].output_desc),
														CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,0,
														&desc[i].algo_desc);
			if(status != CUDNN_STATUS_SUCCESS) return (int)status;
			status = cudnnGetConvolutionForwardWorkspaceSize(*handle, (desc[i].input_desc),
															(desc[i].filter_desc), (desc[i].conv_desc),
															(desc[i].output_desc), desc[i].algo_desc,
															&desc[i].workspace_size);
			if(status != CUDNN_STATUS_SUCCESS) return (int)status;
		} else {
			status = cudnnSetTensor4dDescriptor((desc[i].output_desc), CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, 1, layers[i].fc_layer.size, 1);
			if(status != CUDNN_STATUS_SUCCESS) return (int)status;
			status = cudnnSetTensor4dDescriptor((desc[i].y_desc), CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, 1, layers[i].fc_layer.size, 1);
			if(status != CUDNN_STATUS_SUCCESS) return (int)status;
			status = cudnnSetActivationDescriptor((desc[i].acti_desc), layers[i].fc_layer.activation, CUDNN_NOT_PROPAGATE_NAN, 0.5);
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
					status = cudnnGetTensor4dDescriptor((desc[i - 1].output_desc), &t, &n, &c, &h, &w,
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
			status = cudnnGetFilter4dDescriptor((desc[i].filter_desc), &t, &format, &n,&c,&h,&w);
			cudaMalloc(&desc[i].d_filter, n*c*h*w*sizeof(float));
			if(i==num_layers-1) {
				status = cudnnGetTensor4dDescriptor((desc[i].output_desc), &t, &n, &c, &h, &w,
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
				stat = cudaMalloc(&desc[i].d_weights, (layers[i].fc_layer.input_size)/batch_size*layers[i].fc_layer.size*sizeof(float));
				syslog(LOG_DEBUG, "Memory allocated to d_weights for layer %d PTR to d_weights %p", i, (void *) desc[i].d_weights);
				if(stat != cudaSuccess) return stat;
				stat = cudaMalloc(&desc[i].d_y, batch_size*layers[i].fc_layer.size*sizeof(float));
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
	FILE* fp = fopen("input_post_copy", "w");

	stat = cudaMemcpy(desc[0].d_input, input_image, sizeof(float)*batch_size*IMAGE_WIDTH*IMAGE_HEIGHT, cudaMemcpyHostToDevice);
	
	if(stat != cudaSuccess) {
		syslog(LOG_ERR, "Encountered Error %d when copying input_image to d_input", stat);
		fclose(fp);
		return stat;
	}
	print_to_file(fp, desc[0].d_input, sizeof(float)*batch_size*IMAGE_WIDTH*IMAGE_HEIGHT, "Input_Post_copy", -1);
	fclose(fp);
	for(int i=0; i< num_layers; i++) {
		if(desc[i].valid)  {
			status = cudnnGetFilter4dDescriptor((desc[i].filter_desc), &t, &format, &n,&c,&h,&w);
			if(status != CUDNN_STATUS_SUCCESS) return status;
			stat = cudaMemcpy(desc[i].d_filter, layers[i].conv_layer.filter,
								sizeof(float)*n*c*h*w, cudaMemcpyHostToDevice);
			if(stat != cudaSuccess) return stat;
		} else {
			stat = cudaMemcpy(desc[i].d_weights, layers[i].fc_layer.weights ,
						sizeof(float)*layers[i].fc_layer.input_size*layers[i].fc_layer.size*1/batch_size,
						cudaMemcpyHostToDevice);
			if(stat != cudaSuccess) return stat;
		}
	}
	return 0;

}


struct Status feedforward(cudnnHandle_t* cudnn, cublasHandle_t* handle, struct descriptor* desc, struct layer *layers, int num_layers, int batch_size)
{
	struct Status ff_stat;
	cudnnStatus_t status;
	cublasStatus_t stat;
	float* output_array;
	const float alpha=1, beta=0;
	FILE* fp = fopen("values.txt", "w");
	for(int i=0;i < num_layers;i++) {
        output_array = (i < num_layers-1) ? desc[i+1].d_input:desc[i].d_output;
		if(desc[i].valid) {
				status = cudnnConvolutionForward(*cudnn,&alpha, (desc[i].input_desc), desc[i].d_input,
											(desc[i].filter_desc),desc[i].d_filter, (desc[i].conv_desc),
											 desc[i].algo_desc, desc[i].d_workspace,desc[i].workspace_size,
											 &beta, (desc[i].output_desc), output_array);
				if(status != CUDNN_STATUS_SUCCESS) {
					ff_stat.failure = CUDNN;
					ff_stat.cudnn_stat = status;
					fclose(fp);
					return ff_stat;
				}
		} else {
				assert(desc[i].d_input != NULL);
				assert(desc[i].d_y != NULL );
				assert(desc[i].d_weights != NULL);
				print_to_file(fp, desc[i].d_input, layers[i].fc_layer.input_size, "d_input", i);
				//print_to_file(fp, desc[i].d_weights, layers[i].fc_layer.input_size, "d_weights", i);
				print_to_file(fp, desc[i].d_y, layers[i].fc_layer.size*batch_size, "d_y_before_mult", i);

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
									desc[i].d_y,
									layers[i].fc_layer.size);
				if (stat != CUBLAS_STATUS_SUCCESS) {
									ff_stat.failure = CUBLAS;
									ff_stat.cublas_stat=stat;
									fclose(fp);
									return ff_stat;
				}

				print_to_file(fp, desc[i].d_y, layers[i].fc_layer.size*batch_size, "d_y", i);
				status = cudnnActivationForward(*cudnn, desc[i].acti_desc, &alpha,
												desc[i].y_desc, desc[i].d_y, &beta,
												desc[i].output_desc , output_array);
				if(status != CUDNN_STATUS_SUCCESS) {
								ff_stat.failure = CUDNN;
								ff_stat.layer = i;
								ff_stat.cudnn_stat=status;
								fclose(fp);
								return ff_stat;

				}
				print_to_file(fp, output_array, layers[i].fc_layer.size*batch_size, "output_array ", i);
				printf("Num Layers: %d\n", num_layers);
			}
		}
	ff_stat.failure=NONE;
	fclose(fp);
	return ff_stat;
}



int computecost(float* y, float* yhat, float* ones_vector, int size, cublasHandle_t handle, float* cost) {
	cudaError_t status;
	cublasStatus_t stat;
	int blockSize,gridSize;
	FILE* fp = fopen("output.txt", "w");
	float* h_yhat = (float* )malloc(size*sizeof(float));
	float* h_y = (float* )malloc(size*sizeof(float));
	float* d_result;
	float* h_result = (float*) malloc(size*sizeof(float));
	blockSize = 256;
	gridSize = (int) ceil ((float ) size/blockSize);
	status = cudaMalloc(&d_result, size*sizeof(float));
	if (status != cudaSuccess) { syslog(LOG_ERR, "Allocation of d_result failed"); return status;}
	status = cudaMemcpy(h_yhat, yhat, size*sizeof(float), cudaMemcpyDeviceToHost);
	if (status != cudaSuccess) { syslog(LOG_ERR, "Memcpy of yhat failed with Error code: %d", (int)status); return status;}
	status = cudaMemcpy(h_y, y, size*sizeof(float), cudaMemcpyDeviceToHost);
	if (status != cudaSuccess) { syslog(LOG_ERR, "Memcpy of y failed with Error code: %d", (int)status); return status;}
	 printf("\n\n Printing Data from the GPU: d_y \n\n");
	    print_matrix(h_y, (int)size/32, 32);
	cross_entropy<<<gridSize, blockSize>>>(size, y, yhat, d_result);
    status = cudaDeviceSynchronize();
    if (status != cudaSuccess) { syslog(LOG_ERR, "CudaDeviceSync failed with Error code: %d", (int)status); return status;}
    stat = cublasSdot_v2(handle, size, ones_vector,1, d_result, 1, cost);
    if(stat != CUBLAS_STATUS_SUCCESS ){ syslog(LOG_ERR, "CUBLAS dot product failed with Error code: %d", (int)stat); return stat;}

    status = cudaMemcpy(h_result, d_result, size*sizeof(float), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) { syslog(LOG_ERR, "CudaMemcpy of h_result failed with Error code: %d", (int)status); return status;}

    for (int i=0; i< size;i++) {
    	fprintf(fp, "i: %d yhat: %2.3f y : %2.3f result: %2.3f \n", i, h_yhat[i], h_y[i], h_result[i]);
    	//fprintf(fp, "i: %d yhat: %2.3f y : %2.3f \n", i, yhat[i], y[i]);
    	//fprintf(fp, "HelloWorld\n");
    }
    cudaFree(d_result);
    free(h_result);
    free(h_yhat);
    free(h_y);
    fclose(fp);
    *cost /= size;
	return 0;
}
