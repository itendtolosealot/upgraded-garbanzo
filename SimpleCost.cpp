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

void simpleCostTest() {
	float* h_y;
	float* d_y;
	float* h_yhat;
	float* d_yhat;
	float* h_one_vector;
	float* d_one_vector;
	float cost;
	cublasHandle_t cublas;
	int status;
	int size = 8192;

	cudaMalloc(&d_y, size * sizeof(float));
	cudaMalloc(&d_yhat, size * sizeof(float));
	cudaMalloc(&d_one_vector, size * sizeof(float));

	h_one_vector = (float*)calloc(size, sizeof(float));
	get_matrix(&h_y, size, 1, 1);
	get_matrix(&h_yhat, size, 1, 1);
	for (int i = 0; i < size; i++)
		h_one_vector[i] = 1;
	for (int i = 0; i < 10; i++)
		printf(" %2.3f", h_y[i]);
	printf("\n");

	cudaMemcpy(d_y, h_y, size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_yhat, h_yhat, size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_one_vector, h_one_vector, size * sizeof(float), cudaMemcpyHostToDevice);

	if ((status = (int)cublasCreate(&cublas)) != (int)CUBLAS_STATUS_SUCCESS) {
		syslog(LOG_ERR, "Unable to create CUDA handlers. Terminating the program");
		exit(1);
	}
	computecost(d_y, d_yhat, d_one_vector, size, cublas, &cost);
	free(h_y);
	free(h_yhat);
	free(h_one_vector);
	cudaFree(d_y);
	cudaFree(d_yhat);
	cudaFree(d_one_vector);
	cublasDestroy(cublas);
	printf("Cost is %2.3f\n", cost);
}