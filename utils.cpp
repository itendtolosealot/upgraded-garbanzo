/*
 * utils.c
 *
 *  Created on: Oct 30, 2018
 *      Author: eashvla
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils.h"

void MultiplyCPU(float* A, float* B, float* C, int m, int k, int n) {
	for(int i=0;i< m; i++) {
		for (int j=0; j < n; j++) {
			int sum = 0;
			for (int l=0; l< k; l++) {
				sum += A[m*l+i]*B[m*j+l];
			}
			C[i*m+j]=sum;
		}
	}
}

void  get_matrix(float** mat, int size_x, int size_y, int type ) {

        float* matrix;
        matrix = (float*) malloc(size_x * size_y*sizeof(float));
        for (int i=0;i<size_x*size_y;i++) {
        	if (type == 1)
            	matrix[i] = (rand()*1.0)/(RAND_MAX*1.0);
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

 void create_output_arrays_in_gpu(float** h_y, float** d_y, float** h_one_vec, float** d_one_vec, int size_x, int size_y) {
		get_matrix(h_y, size_x, size_y,1);
		cudaMalloc(d_y, size_x*size_y*sizeof(float));
		cudaMemcpy(d_y, h_y, size_x*size_y*sizeof(float), cudaMemcpyHostToDevice);
		float* h_one = (float*) calloc(size_x*size_y, sizeof(float));
		for (int i=0; i< size_x*size_y; i++)
			h_one[i]= 1;
		*h_one_vec = h_one;
		cudaMalloc(d_one_vec,size_x*size_y*sizeof(float));
		cudaMemcpy(d_one_vec, h_one_vec, size_x*size_y*sizeof(float), cudaMemcpyHostToDevice);
 }

void delete_output_arrays_from_gpu(float* h_y, float* d_y,float* h_one_vec, float* d_one_vec) {
	free(h_y);
	free(h_one_vec);
	cudaFree(d_y);
	cudaFree(d_one_vec);
}

