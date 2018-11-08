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

void NNbyCPU(struct layer* layers, int num_layers, float* input_image, float* y, int batch_size, float* cost) {
	//FILE* fp = fopen("values_cpu.txt", "w");
	float** output = (float**) calloc(num_layers, sizeof(float*));
	float* input;
	for(int i=0; i< num_layers;i++) {
		output[i] = (float*) mkl_malloc(layers[i].fc_layer.size*batch_size*sizeof(float), 32);
	}

	for(int i=0; i< num_layers;i++) {
		input = (i==0) ? input_image:output[i-1];
		assert(input != NULL);
		int m = layers[i].fc_layer.size;
		int k = (layers[i].fc_layer.input_size)/batch_size;
		int n = batch_size;
		MultiplyCPU(layers[i].fc_layer.weights,input,output[i], m, k, n);
		sigmoidCPU(output[i], m*n);
	}
	//fclose(fp);
	computeCostCPU(y, output[num_layers-1], layers[num_layers-1].fc_layer.size*batch_size, cost);

	for(int i=0; i< num_layers;i++) {
		assert(output[i] != NULL);
		mkl_free(output[i]);
	}
	free(output);
}


void MultiplyCPU(float* A, float* B, float* C, int m, int k, int n) {
	/*for(int i=0;i< m; i++) {
		for (int j=0; j < n; j++) {
			float sum = 0;
			for (int l=0; l< k; l++) {
				// sum += A[i][l]*B[l][j];
				// Column Major Formula
				sum += A[m*l+i]*B[k*j+l];
				//Row Major Formula
				//sum += A[m*i+l]*B[l*n+j];
			}
			//Column Major Formula
			C[j*m+i]=sum;
			//Row Major Formula
			//printf("i: %d j: %d m: %d n: %d k: %d value: %2.3f", i, j, m, k, n, sum);
			//C[i*n+j] = sum;
		}
	}*/
	cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, A, k, B, n, 0, C, n);
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
        matrix = (float*) mkl_malloc(size_x * size_y*sizeof(float), 32);
        for (int i=0;i<size_x*size_y;i++) {
        	if (type == 1)
            	matrix[i] = ((rand()*1.0)/(RAND_MAX)-0.5)/2.0;
            else
            	matrix[i] = 0;
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

 int create_output_arrays_in_gpu(float** h_y, float** d_y, float** h_one_vec, float** d_one_vec, int size_x, int size_y) {
	 	cudaError_t status;
		get_matrix(h_y, size_x, size_y,1);
		//print_matrix(*h_y, size_x, size_y);
		status = cudaMalloc(d_y, size_x*size_y*sizeof(float));
		if (status != cudaSuccess) { syslog(LOG_ERR, "Allocation of d_y failed with error code %d", status); return status;}

		status = cudaMemcpy(*d_y, *h_y, size_x*size_y*sizeof(float), cudaMemcpyHostToDevice);
		if (status != cudaSuccess) { syslog(LOG_ERR, "Memcpy of h_y to d_y failed with error code %d", status); return status;}
		float* h_one = (float*) calloc(size_x*size_y, sizeof(float));
		for (int i=0; i< size_x*size_y; i++)
			h_one[i]= 1;
		*h_one_vec = h_one;
		status = cudaMalloc(d_one_vec,size_x*size_y*sizeof(float));
		if (status != cudaSuccess) { syslog(LOG_ERR, "Allocation of d_one_vec failed with error code %d", status); return status;}
		status = cudaMemcpy(*d_one_vec, *h_one_vec, size_x*size_y*sizeof(float), cudaMemcpyHostToDevice);
		if (status != cudaSuccess) { syslog(LOG_ERR, "Memcpy of h_onevec to d_onevec failed with error code %d", status); return status;}
		return 0;
 }

int  delete_output_arrays_from_gpu(float* h_y, float* d_y,float* h_one_vec, float* d_one_vec) {
	cudaError_t status;
	free(h_y);
	free(h_one_vec);
	status = cudaFree(d_y);
	if(status != cudaSuccess) return (int) status;
	status = cudaFree(d_one_vec);
	if(status != cudaSuccess) return (int)status;
}

int destroy_layers(struct layer* layers, float* input_image, int num_layers) {
	for (int i = 0; i < num_layers; i++) {
		if (layers[i].type == FULLYCONNECTED) {
			mkl_free(layers[i].fc_layer.weights);
		}
		else if (layers[i].type == CONVOLUTION) {
			mkl_free(layers[i].conv_layer.filter);
		}
	}
	mkl_free(input_image);
	free(layers);
}