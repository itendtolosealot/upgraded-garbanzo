/*
 * utils.h
 *
 *  Created on: Oct 30, 2018
 *      Author: eashvla
 */

#ifndef UTILS_H_
#define UTILS_H_
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
void  get_matrix(float** mat, int size_x, int size_y, int type ) ;
void print_matrix(float* Result, int size_x, int size_y);
void MultiplyCPU(float* A, float* B, float* C, float* X, int m, int k, int n);
int create_output_arrays_in_gpu(float** h_y, float** d_y, float** h_one_vec, float** d_one_vec, int size_x, int size_y);
int delete_output_arrays_from_gpu(float* h_y, float* d_y,float* h_one_vec, float* d_one_vec) ;
void print_to_file(FILE* fp, float* x, int size, const char* varName, int layer_id);
void NNbyCPU(struct layer* layers, int num_layers, float* input_image, float* y, int batch_size, float* cost);
void sigmoidCPU(float* A, int size);
void computeCostCPU(float* y, float* yhat, int size, float* cost);
int destroy_layers(struct layer* layers, float* input_image, int num_layers);
double gigaFlop(struct layer* layers, int num_layers, int batch_size);
float* replicate_bias_for_batch(int size, int batch_size, float* bias);
#endif /* UTILS_H_ */
