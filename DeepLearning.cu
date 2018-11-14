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
#include <syslog.h>
#include "DeepLearning.h"
#include "utils.h"

/* We calculate the sum_exponents for each example in the batch, and use that value to calculate the cross entropy*/
/* sum_exponents is calculated as the sum of the exponents for a given example, i.e., sum_exp = \sum_{i=0}^{output_size} exp(yhat);*/
/* In cross_entropy kernel, the input exp_yhat represents exp(yhat[i]) for all i in {0..., output_size*batch_size} */

__global__ void cross_entropy(int batch_size, int output_size, float* y, float* exp_yhat, float* sum_exponents)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int sum_exp_index = i / output_size;
  float res = 0;
  if (i < batch_size*output_size) {
	  if ((exp_yhat[i] == 0) || (exp_yhat[i] == 1))
			  res = 0;
		  else
			  res = log(1 - exp_yhat[i] / (sum_exponents[sum_exp_index]))*y[i] + log(exp_yhat[i] / (sum_exponents[sum_exp_index]))*(1 - y[i]);
	  }
	  exp_yhat[i] = res;
}

/* We calculate the exponent of the output for every output*/
__global__ void softmax(int array_size, float* yhat)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	float res = 0;
	if (i < array_size) {
		res = exp(yhat[i]);
		yhat[i] = res;
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

int destroy_descriptors (struct descriptor* desc, struct cost_descriptor cost, int num_layers) {
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
			if (desc[i].d_bias != NULL) cudaFree(desc[i].d_bias);
			cudnnDestroyTensorDescriptor(desc[i].y_desc);
			cudnnDestroyTensorDescriptor(desc[i].output_desc);
			cudnnDestroyActivationDescriptor(desc[i].acti_desc);
		}

		if(desc[i].d_input != NULL) cudaFree(desc[i].d_input);
		if(desc[i].d_filter != NULL) cudaFree(desc[i].d_filter);
		if(desc[i].d_workspace != NULL) cudaFree(desc[i].d_workspace);
		if (cost.d_dout != NULL) cudaFree(cost.d_dout);
		if (cost.d_out != NULL) cudaFree(cost.d_out);
		if (cost.d_one_vec != NULL) cudaFree(cost.d_one_vec);
		if (cost.d_y != NULL) cudaFree(cost.d_y);
		if (cost.d_yhat != NULL) cudaFree(cost.d_yhat);
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

cudaError_t allocate_memory_cost_desc(struct cost_descriptor cost, int size) {
	cudaError_t stat;
	stat = cudaMalloc(&cost.d_out, size*sizeof(float));
	if (stat != cudaSuccess) return stat;
	stat = cudaMalloc(&cost.d_dout, size * sizeof(float));
	if (stat != cudaSuccess) return stat;
	stat = cudaMalloc(&cost.d_yhat, size * sizeof(float));
	if (stat != cudaSuccess) return stat;
	stat = cudaMalloc(&cost.d_y, size * sizeof(float));
	if (stat != cudaSuccess) return stat;
	stat = cudaMalloc(&cost.d_one_vec, size * sizeof(float));
	if (stat != cudaSuccess) return stat;
	
	cost.h_y = (float*) mkl_malloc(size * sizeof(float), 64);
	if (cost.h_y == NULL) {
		syslog(LOG_ERR, "Unable to allocate memory to h_y");
		return 1;
	}
	cost.h_one_vec = (float*)mkl_malloc(size * sizeof(float), 64);
	if (cost.h_one_vec == NULL) {
		syslog(LOG_ERR, "Unable to allocate memory to h_y");
		return 1;
	}
	return cudaSuccess;
}

int allocate_memory(struct descriptor* desc, struct cost_descriptor cost, struct layer* layers, int num_layers, int batch_size) {
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
				if (status != CUDNN_STATUS_SUCCESS) { return (int)status; }
				stat = cudaMalloc(&desc[i].d_input, n*c*h*w*sizeof(float));
				if (stat != cudaSuccess) { return stat; }
			}
			status = cudnnGetFilter4dDescriptor((desc[i].filter_desc), &t, &format, &n,&c,&h,&w);
			cudaMalloc(&desc[i].d_filter, n*c*h*w*sizeof(float));
			if(i==num_layers-1) {
				status = cudnnGetTensor4dDescriptor((desc[i].output_desc), &t, &n, &c, &h, &w, NULL, NULL, NULL, NULL);
				if(status != CUDNN_STATUS_SUCCESS) return (int)status;
				stat = allocate_memory_cost_desc(cost, n*c*h*w);
				if (stat != cudaSuccess) {
					syslog(LOG_ERR, "Cost struct memory allocation failed with Error %d", stat);
					return stat;
				}
			}
			stat = cudaMalloc(&desc[i].d_workspace,desc[i].workspace_size);
			if (stat != cudaSuccess) {
				syslog(LOG_ERR, "workspace memory allocation failed with Error %d", stat);
				return stat;
			}

		} else {
				stat = cudaMalloc(&desc[i].d_input, layers[i].fc_layer.input_size*sizeof(float));
				if(stat != cudaSuccess) return stat;
				stat = cudaMalloc(&desc[i].d_weights, (layers[i].fc_layer.input_size)/batch_size*layers[i].fc_layer.size*sizeof(float));
				syslog(LOG_DEBUG, "Memory allocated to d_weights for layer %d PTR to d_weights %p", i, (void *) desc[i].d_weights);
				if(stat != cudaSuccess) return stat;
				stat = cudaMalloc(&desc[i].d_bias, batch_size*layers[i].fc_layer.size*sizeof(float));
				if(stat != cudaSuccess) return stat;
				if(i==num_layers-1) {
					stat = allocate_memory_cost_desc(cost, batch_size*layers[i].fc_layer.size);
					if (stat != cudaSuccess) {
						syslog(LOG_ERR, "Cost struct memory allocation failed with Error %d", stat);
						return stat;
					}
				}
		}


	}
	return 0;
}

int copy_input_to_device(struct descriptor* desc, struct cost_descriptor cost, struct layer* layers, int num_layers, float* input_image, int batch_size)
{
	cudnnStatus_t status;
	cudaError_t stat;
	cudnnDataType_t t;
	cudnnTensorFormat_t format;
	int n,c,h,w;

	stat = cudaMemcpy(desc[0].d_input, input_image, sizeof(float)*batch_size*IMAGE_WIDTH*IMAGE_HEIGHT, cudaMemcpyHostToDevice);
	
	if(stat != cudaSuccess) {
		syslog(LOG_ERR, "Encountered Error %d when copying input_image to d_input", stat);
		return stat;
	}

	int size_x
	int size_y;
	if (desc[num_layers - 1].valid) {
		status = cudnnGetTensor4dDescriptor((desc[num_layers - 1].output_desc), &t, &n, &c, &h, &w, NULL, NULL, NULL, NULL);
		if (status != 0) {
			syslog(LOG_ERR, "Error while determining Output vec size. Terminating the program");
			return status;
		}
		size_x = n * c;
		size_y = h * w;
	}
	else {
		size_y = layers[num_layers - 1].output_size;
		size_x = batch_size;
	}

	get_matrix(&cost.h_y, size_x, size_y, 2);
	for(int i = 0; i < size_x*size_y; i++) {
		cost.h_one_vec[i] = 1;
	}

	stat = cudaMemcpy(cost.d_one_vec, cost.h_one_vec, size_x*size_y * sizeof(float), cudaMemcpyHostToDevice);
	if (stat != cudaSuccess) {
		syslog(LOG_ERR, "Error while copying one vector to the device.");
		return stat;
	}
	stat = cudaMemcpy(cost.d_y, cost.h_y, size_x*size_y * sizeof(float), cudaMemcpyHostToDevice);
	if (stat != cudaSuccess) {
		syslog(LOG_ERR, "Error while copying y vector to the device.");
		return stat;
	}

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
			for(int j=0; j < batch_size; j++) {
				float* d_bias_matrix = desc[i].d_bias;
				stat = cudaMemcpy((d_bias_matrix+j*layers[i].fc_layer.size), layers[i].fc_layer.bias,sizeof(float)*layers[i].fc_layer.size,cudaMemcpyHostToDevice);
				if(stat != cudaSuccess) return stat;
			}

		}
	}
	return 0;

}


struct Status feedforward(cudnnHandle_t* cudnn, cublasHandle_t* handle, struct descriptor* desc, struct layer *layers, int num_layers, int batch_size)
{
	struct Status ff_stat;
	cudnnStatus_t status;
	cublasStatus_t stat;
	cudaError_t cuda_stat;
	float* output_array;
	const float alpha=1, beta=0;
//  struct timeval start_timeval, end_timeval;

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
					ff_stat.cublas_stat = CUBLAS_STATUS_SUCCESS;
					return ff_stat;
				}
		} else {
				assert(desc[i].d_input != NULL);
				assert(desc[i].d_y != NULL );
				assert(desc[i].d_weights != NULL);
				//gettimeofday(&start_timeval, NULL);
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
									ff_stat.cudnn_stat = CUDNN_STATUS_SUCCESS;
									ff_stat.cuda_stat = cudaSuccess;
									return ff_stat;
				}

				cuda_stat = cudaDeviceSynchronize();
				if(cuda_stat != cudaSuccess) {
					ff_stat.failure = CUDA;
					ff_stat.cublas_stat=CUBLAS_STATUS_SUCCESS;
					ff_stat.cudnn_stat = CUDNN_STATUS_SUCCESS;
					ff_stat.cuda_stat = cuda_stat;
					return ff_stat;
				}

				stat = cublasSaxpy(*handle, layers[i].fc_layer.size*batch_size, &alpha, desc[i].d_bias, 1, desc[i].d_y, 1);
				if (stat != CUBLAS_STATUS_SUCCESS) {
					ff_stat.failure = CUBLAS;
					ff_stat.cublas_stat=stat;
					ff_stat.cudnn_stat = CUDNN_STATUS_SUCCESS;
					ff_stat.cuda_stat = cudaSuccess;
					return ff_stat;
				}

				cuda_stat = cudaDeviceSynchronize();
				if(cuda_stat != cudaSuccess) {
					ff_stat.failure = CUDA;
					ff_stat.cublas_stat=CUBLAS_STATUS_SUCCESS;
					ff_stat.cudnn_stat = CUDNN_STATUS_SUCCESS;
					ff_stat.cuda_stat = cuda_stat;
					return ff_stat;
				}

				/* gettimeofday(&end_timeval, NULL);
				float msec_timeval;
				float flop = 2.0*layers[i].fc_layer.size * layers[i].fc_layer.input_size;
				msec_timeval = (end_timeval.tv_sec - start_timeval.tv_sec)*1000.0 + (end_timeval.tv_usec - start_timeval.tv_usec)*1.0/1000.0;
				printf("GFlops using GPU time in MatMul Layer %d Muls: %2.3f is %2.3f\n", i, flop*1e-9 , flop*1.0*1e-6/(msec_timeval));
				gettimeofday(&start_timeval, NULL);*/

				status = cudnnActivationForward(*cudnn, desc[i].acti_desc, &alpha,
												desc[i].y_desc, desc[i].d_y, &beta,
												desc[i].output_desc , output_array);
				cuda_stat = cudaDeviceSynchronize();
				if(cuda_stat != cudaSuccess) {
									ff_stat.failure = CUDA;
									ff_stat.cublas_stat=CUBLAS_STATUS_SUCCESS;
									ff_stat.cudnn_stat = CUDNN_STATUS_SUCCESS;
									ff_stat.cuda_stat = cuda_stat;
									return ff_stat;
				}

				/*
				gettimeofday(&end_timeval, NULL);
				flop = 2.0*layers[i].fc_layer.size * batch_size;
				msec_timeval = (end_timeval.tv_sec - start_timeval.tv_sec)*1000.0 + (end_timeval.tv_usec - start_timeval.tv_usec)*1.0/1000.0;
				printf("GfLops using GPU Time at Activation Layer %d  is %2.3f\n", i, flop*1e-6/msec_timeval);
				*/
				if(status != CUDNN_STATUS_SUCCESS) {
								ff_stat.failure = CUDNN;
								ff_stat.layer = i;
								ff_stat.cudnn_stat=status;
								ff_stat.cublas_stat = CUBLAS_STATUS_SUCCESS;
								ff_stat.cuda_stat = cudaSuccess;
								return ff_stat;

				}
			}
		}
	ff_stat.failure=NONE;
	return ff_stat;
}


struct Status feedback(cudnnHandle_t* cudnn, cublasHandle_t* handle, struct descriptor* desc, struct cost_descriptor cost, struct layer *layers, int num_layers, int batch_size) {
	struct Status ff_stat;
	cudnnStatus_t status;
	cublasStatus_t stat;
	stat = CUBLAS_STATUS_SUCCESS;
	status= CUDNN_STATUS_SUCCESS;
	float alpha = 1.0;
	float beta = 0.0;

	stat = cublasSaxpy(*handle, layers[i].fc_layer.size*batch_size, &alpha, desc[i].d_bias, 1, desc[i].d_y, 1);
	for (int i = num_layers - 1; i >= 0; i--) {
		if (desc[i].valid) {
			float *dout = (i == num_layers - 1) ? desc[i].d_dout : desc[i + 1].d_din;
			status = cudnnConvolutionBackwardData(*handle, &alpha, desc[i].filter_desc, desc[i].d_filter, desc[i].dout_desc, dout,
													desc[i].conv_desc, layers[i].conv_layer.algorithm, desc[i].d_workspace, desc[i].workspace_size,
													&beta, desc[i].din_desc, desc[i].d_din);
			if (status != CUDNN_STATUS_SUCCESS) {
				return status;
			}
			status = cudnnConvolutionBackwardFilter(*handle, &alpha, desc[i].input_desc, desc[i].d_input,
													desc[i].dout_desc, dout, desc[i].conv_desc, layers[i].conv_layer.algorithm
													desc[i].d_workspace, desc[i].workspace_size, &beta, desc[i].dfilter_desc, desc[i].d_df);
			if (status != CUDNN_STATUS_SUCCESS) {
				return status;
			}
		}
		else {
			return ff_stat;
		}
	}

	return ff_stat;

}



int computecost(float* y, float* yhat, float* ones_vector, int batch_size, int output_size, cublasHandle_t handle, float* cost) {
	cudaError_t status;
	cublasStatus_t stat;
	float* sum_exponents = cudaMalloc(sizeof(float)*batch_size);
	float alpha = 1;
	float beta = 0;
	int blockSize,gridSize;
	blockSize = 1024;
	gridSize = (int) ceil ((float ) batch_size*output_size/(blockSize));
	/* Softmax on every output. The result is stored in yhat itself. */
	softmax << <gridSize, blockSize >> > (batch_size*output_size, yhat);
	status = cudaDeviceSynchronize();
	if (status != cudaSuccess) { syslog(LOG_ERR, "CudaDeviceSync failed with Error code: %d during Kernel run of softmax", (int)status); return status; }
	
	/* Matrix mul to find \sum_{i=0}^{output_size} yhat[i]. This will give the sum of exponents for a given exaomple*/
	stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, batch_size, output_size, &alpha, ones_vector, 1, yhat, output_size, &beta, sum_exponents, 1);
	if (stat != CUBLAS_STATUS_SUCCESS) { syslog(LOG_ERR, "CUBLAS matrix mult to compute sum_exponents failed with Error code: %d ", (int)stat); return stat;}

	/* Calculating cross entropy knowing the sum of exponents*/
	cross_entropy<<<gridSize, blockSize>>>(batch_size, output_size, y, yhat, sum_exponents);
    status = cudaDeviceSynchronize();
    if (status != cudaSuccess) { syslog(LOG_ERR, "CudaDeviceSync failed with Error code: %d during Kernel run of cross_entropy", (int)status); return status;}

	/* Dot product to compute the sum of all the log properties*/
    stat = cublasSdot_v2(handle, batch_size*output_size, ones_vector,1, yhat, 1, cost);
    status = cudaDeviceSynchronize();
    if (status != cudaSuccess) { syslog(LOG_ERR, "CudaDeviceSync failed with Error code: %d in cublasSdot", (int)status); return status;}
    if(stat != CUBLAS_STATUS_SUCCESS ){ syslog(LOG_ERR, "CUBLAS dot product failed with Error code: %d", (int)stat); return stat;}
	*cost /= (batch_size);
	return 0;
}
