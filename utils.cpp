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
            	matrix[i] = rand()/RAND_MAX;
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


