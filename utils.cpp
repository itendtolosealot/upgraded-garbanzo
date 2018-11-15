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

float* replicate_bias_for_batch(int size, int batch_size, float* bias) {
	float* replicated_bias = (float *) mkl_malloc(sizeof(float)*size*batch_size, 64);
	for(int i=0; i < batch_size;i++) {
		memcpy(replicated_bias + i*size, bias, size*sizeof(float));
	}
	return replicated_bias;
}

void NNbyCPU(struct layer* layers, int num_layers, float* input_image, float* y, int batch_size, float* cost) {
	//FILE* fp = fopen("values_cpu.txt", "w");
	float** output = (float**) mkl_calloc(num_layers, sizeof(float*), 64);
	float** bias = (float**) mkl_calloc(num_layers, sizeof(float*), 64);
	float* input;
	float* sum_exponents = (float*)mkl_calloc(batch_size, sizeof(float));
	float* exp_output = (float*)mkl_calloc(batch_size*layers[num_layers - 1].fc_layer.size, sizeof(float));

	for(int i=0; i< num_layers;i++) {
		output[i] = (float*) mkl_malloc(layers[i].fc_layer.size*batch_size*sizeof(float), 64);
		bias[i] = replicate_bias_for_batch(layers[i].fc_layer.size, batch_size, layers[i].fc_layer.bias);
	}

	for (int i = 0; i < num_layers; i++) {
		if (layers[i].type = FULLYCONNECTED) {
			input = (i == 0) ? input_image : output[i - 1];
			assert(input != NULL);
			int m = layers[i].fc_layer.size;
			int k = (layers[i].fc_layer.input_size) / batch_size;
			int n = batch_size;
			MultiplyCPU(layers[i].fc_layer.weights, input, output[i], bias[i], m, k, n);
			sigmoidCPU(output[i], m*n);
		}
	}
	softmaxCPU(output[num_layers-1], exp_output, sum_exponents, m, n, k);
	//fclose(fp);
	computeCostCPU(y, output[num_layers-1], layers[num_layers-1].fc_layer.size*batch_size, cost);

	for(int i=0; i< num_layers;i++) {
		assert(output[i] != NULL);
		assert(bias[i] != NULL);
		mkl_free(output[i]);
		mkl_free(bias[i]);
	}
	if (sum_exponents != NULL) mkl_free(sum_exponents);
	if (exp_output != NULL) mkl_free(exp_output);
	if (output != NULL) mkl_free(output);
	if (bias != NULL) mkl_free(bias);
}
void softmaxCPU(float* out, float* exp_out, float* sum_exp, int m, int n, int k) {
	float* one_vector = (float*)mkl_malloc(sizeof(float)*k);
	for (int i = 0; i < k; i++) {
		one_vector[i] = 1;
	}
	for (int i = 0; i < size; i++) {
		exp_out[i] = (float)exp(out[i]);
	}
	cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, exp_out, m, one_vector, k, 0, sum_exp, m);
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

void computeCostCPU(float* y, float* yhat, int size, float* cost) {

	*cost = 0;
	for (int i=0;i< size; i++) {
		if((yhat[i] != 0) && (yhat[i] != 1)) {
		*cost += log(1-yhat[i])*y[i] + log(yhat[i])*(1-y[i]);
		}
	}
	*cost /= size;
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

void  print_to_file(FILE* fp, float* x, int size, const char* varName, int layer_id) {
	cudaError_t error;
	if(fp == NULL ){
		syslog(LOG_ERR, "File pointer to print arrays is NULL. Terminating the program");
		exit(1);
	}
	float* h_y = (float* ) malloc(sizeof(float)*size);
	error = cudaMemcpy(h_y, x, size*sizeof(float), cudaMemcpyDeviceToHost);
	if(error != cudaSuccess) {
					syslog(LOG_ERR, "Copying from %s to host caused Error %d ", varName, error);
					free(h_y);
					fclose(fp);
					exit(1);
	}

	for (int i= 0; i< size; i++) {
		fprintf(fp, "Var %s in Layer %d Id: %d val: %2.3f \n", varName, layer_id, i, h_y[i]);
	}
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
}

double gigaFlop(struct layer* layers, int num_layers, int batch_size) {
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

