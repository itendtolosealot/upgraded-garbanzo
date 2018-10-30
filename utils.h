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
void  get_matrix(float** mat, int size_x, int size_y, int type ) ;
void print_matrix(float* Result, int size_x, int size_y);
void MultiplyCPU(float* A, float* B, float* C, int m, int k, int n);

#endif /* UTILS_H_ */