/*
 * utils.c
 *
 *  Created on: Oct 30, 2018
 *      Author: eashvla
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <syslog.h>
#include <math.h>
#include <assert.h>
#include "utils.h"
#include "mkl.h"

void replicate_bias_for_batch(int size, int batch_size, float* bias_src, float** bias_dest) {
	float* replicate = (float *) mkl_malloc(sizeof(float)*size*batch_size, 64);
	for(int i=0; i < batch_size;i++) {
		memcpy(replicate + i*size, bias_src, size*sizeof(float));
	}
	*bias_dest = replicate;
}

void NNbyCPU(struct layer* layers, int num_layers, float* input_image, float* y, int batch_size, float* cost) {
	//FILE* fp = fopen("out.txt", "w");
	float** output = (float**) mkl_calloc(num_layers, sizeof(float*), 64);
	float** bias = (float**) mkl_calloc(num_layers, sizeof(float*), 64);
	float* input;
	float* sum_exponents = (float*)mkl_calloc(batch_size, sizeof(float), 64);
	float* exp_output = (float*)mkl_calloc(batch_size*layers[num_layers - 1].fc_layer.size, sizeof(float), 64);
	int m,n,k;
	for(int i=0; i< num_layers;i++) {
		output[i] = (float*) mkl_malloc(layers[i].fc_layer.size*batch_size*sizeof(float), 64);
		syslog(LOG_DEBUG, "Allocated %lu bytes to Output Array of Layer %d at Pointer %p", (int) layers[i].fc_layer.size*batch_size*sizeof(float), i, output[i]);
		replicate_bias_for_batch(layers[i].fc_layer.size, batch_size, layers[i].fc_layer.bias, &bias[i]);
	}

	for (int i = 0; i < num_layers; i++) {
		if (layers[i].type = FULLYCONNECTED) {
			input = (i == 0) ? input_image : output[i - 1];
			assert(input != NULL);
			m = layers[i].fc_layer.size;
			k = (layers[i].fc_layer.input_size) / batch_size;
			n = batch_size;
			MultiplyCPU(layers[i].fc_layer.weights, input, output[i], bias[i], m, k, n);
			sigmoidCPU(output[i], m*n);
			syslog(LOG_DEBUG, "Layer %d m: %d n: %d k: %d", i, m, n,k);
		}
	}
	softmaxCPU(output[num_layers-1], exp_output, sum_exponents, m, n);
	//print_to_file(fp, sum_exponents,  batch_size, "h_sum_exp", num_layers,1);
	computeCostCPU(y, exp_output, sum_exponents, layers[num_layers-1].fc_layer.size, layers[num_layers-1].fc_layer.size*batch_size, cost);

	//fclose(fp);
	for(int i=0; i< num_layers;i++) {
		assert(output[i] != NULL);
		assert(bias[i] != NULL);
		syslog(LOG_DEBUG, "Freeing Bias of Layer %d", i);
		mkl_free(bias[i]);
		syslog(LOG_DEBUG, "Freeing output of Layer %d pointer %p", i, output[i]);
		mkl_free(output[i]);

	}
	if (sum_exponents != NULL) mkl_free(sum_exponents);
	if (exp_output != NULL) mkl_free(exp_output);
	if (output != NULL) mkl_free(output);
	if (bias != NULL) mkl_free(bias);
}
void softmaxCPU(float* out, float* exp_out, float* sum_exp, int m, int n) {
	float* one_vector = (float*)mkl_malloc(sizeof(float)*m, 64);
	for (int i = 0; i < m; i++) {
		one_vector[i] = 1;
	}
	for (int i = 0; i < m*n; i++) {
		exp_out[i] = (float)exp(out[i]);
	}
	cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 1, n, m, 1.0, one_vector, 1, exp_out, m, 0, sum_exp, 1);
	mkl_free(one_vector);
}


void MultiplyCPU(float* A, float* B, float* C, float* X,  int m, int k, int n) {
	cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, A, m, B, k, 0, C, m);
	cblas_saxpy(m*n, 1.0, X, 1, C, 1);
}

void sigmoidCPU(float* A, int size) {
	for (int i=0; i < size;i++) {
		A[i] = (float) 1.0/(1+exp(-(float) A[i]));
	}
}

void computeCostCPU(float* y, float* yhat, float* sum_exp, int output_size, int size, float* cost) {

	*cost = 0;
	int index =0;
	for (int i=0;i< size; i++) {
		index = i/output_size;
		if((yhat[i] != 0) && (yhat[i] != 1)) {
		if(yhat[i] < 0 || sum_exp[index] < 0) {
			syslog(LOG_ERR, "yhat[%d]: %2.3f sum_exp[%d]: %2.3f", i, yhat[i], index, sum_exp[index]);
		}
		*cost += log(1-yhat[i]/sum_exp[index])*y[i] + log(yhat[i]/sum_exp[index])*(1-y[i]);
		}
	}
	*cost /= (size/output_size);
	*cost = -(*cost);
}

void  get_matrix(float** mat, int size_x, int size_y, int type ) {

        float* matrix;
		int index;
        matrix = (float*) mkl_malloc(size_x * size_y*sizeof(float), 64);
        for (int i=0;i<size_x*size_y;i++) {
			if (type == 0)
				matrix[i] = 0;
			else if(type ==1)
            	matrix[i] = ((rand()*1.0)/(RAND_MAX)-0.5)/2.0;          
        }
		if (type == 2) {
			for (int i = 0; i < size_x; i++) {
				index = rand() % size_y;
				for (int j = 0; j < size_y; j++)
					if (j == index)
						matrix[i*size_y + j] = 1;
					else
						matrix[i*size_y + j] = 0;
			}
		}
        *mat = matrix;
}


void print_matrix(float* Result, int size_x, int size_y) {
    printf("\n");
    for (int j=0;j< size_y;j++) {
        for (int i=0;i< size_x; i++){
                        printf("%.2f ", Result[i*size_x+j]);
        }
        printf("\n");
    }
}

void  print_to_file(FILE* fp, float* x, int size, const char* varName, int layer_id, int host_mem) {
	cudaError_t error;
	if(fp == NULL ){
		syslog(LOG_ERR, "File pointer to print arrays is NULL. Terminating the program");
		exit(1);
	}
	float* h_y;
	if(!host_mem) {
	h_y = (float* ) malloc(sizeof(float)*size);
	error = cudaMemcpy(h_y, x, size*sizeof(float), cudaMemcpyDeviceToHost);
	if(error != cudaSuccess) {
					syslog(LOG_ERR, "Copying from %s to host caused Error %d ", varName, error);
					free(h_y);
					fclose(fp);
					exit(1);
	}
	} else {
		h_y = x;
	}
	for (int i= 0; i< size; i++) {
		fprintf(fp, "Var %s in Layer %d Id: %d val: %2.3f \n", varName, layer_id, i, h_y[i]);
	}
	if(!host_mem)
		free(h_y);
}

int destroy_layers(struct layer* layers, float* input_image, int num_layers) {
	for (int i = 0; i < num_layers; i++) {
		if (layers[i].type == FULLYCONNECTED) {
			mkl_free(layers[i].fc_layer.weights);
			mkl_free(layers[i].fc_layer.bias);
		}
		else if (layers[i].type == CONVOLUTION) {
			mkl_free(layers[i].conv_layer.filter);
		}
	}
	mkl_free(input_image);
	free(layers);
	return 0;
}

double gigaFlop(struct layer* layers, int num_layers, int batch_size, int IMAGE_WIDTH, int IMAGE_HEIGHT) {
	double gFlops = 0;
	int input_size = IMAGE_WIDTH*IMAGE_HEIGHT*batch_size;

	for(int i=0; i<num_layers;i++) {
		if(layers[i].type==FULLYCONNECTED) {
			gFlops += 2.0*(double)(layers[i].fc_layer.input_size)*(double)(layers[i].fc_layer.size);
			gFlops += (double) layers[i].fc_layer.size*batch_size;
			if(i==num_layers-1)
				gFlops+= 5.0*(double) layers[i].fc_layer.size*batch_size;
		} else {
			if(i!= 0)
				input_size = (input_size - layers[i-1].conv_layer.filter_size + 2*layers[i-1].conv_layer.padding)/layers[i-1].conv_layer.stride + 1;
				gFlops += 2.0* (double)(layers[i].conv_layer.filter_size*layers[i].conv_layer.filter_size*((input_size - layers[i].conv_layer.filter_size + 2*layers[i].conv_layer.padding)/layers[i].conv_layer.stride +1));
		}
	}
	return gFlops;
}

void populate_error_status(struct Status* ff_stat, int error_type, int error, int layer) {
	ff_stat->failure = (FailureType) error_type;
	ff_stat->layer = layer;
	switch (error_type) {
	case 0: ff_stat->cuda_stat = (cudaError_t) error;
	case 1: ff_stat->cublas_stat = (cublasStatus_t) error;
	case 2: ff_stat->cudnn_stat = (cudnnStatus_t)error;
	}
}

const char* cublasGetErrorString(cublasStatus_t status)
{
    switch(status)
    {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
    }
    return "unknown error";
}
