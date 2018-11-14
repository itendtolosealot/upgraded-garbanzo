/*
 * main.c
 *
 *  Created on: Oct 30, 2018
 *      Author: eashvla
 */
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
#include <sys/time.h>
constexpr auto FP = "./layers.info";

int main(int argc, char **argv) {
	FILE* fp;
	int num_layers;
	struct layer* layers = NULL;
	struct descriptor* desc = NULL;
	LayerType type;
	cudnnHandle_t cudnn;
	cublasHandle_t cublas;
	int status;
	struct cost_descriptor cost;
	struct Status ff_stat;
	int batch_size;
	float* input_image;
	struct cost_descriptor cost;
	float cost;
	float* test;
	int n, c, h, w;
	int num_turns;
	cudnnDataType_t t;

	setlogmask (LOG_UPTO (LOG_ERR));
	openlog ("deep-learning", LOG_CONS | LOG_PID | LOG_NDELAY, LOG_USER);
	if ((fp=fopen(FP, "r"))==NULL) {
		syslog (LOG_ERR, "Unable to find the layer information file. Terminating the program");
		exit(1);
	} else {
		syslog(LOG_DEBUG, "Obtained file Handler for the file %s", FP);
	}

	fscanf(fp, "%d %d %d \n", &num_layers, &batch_size, &num_turns);
	syslog(LOG_DEBUG, "Number of Layers: %d", num_layers);
	if (num_layers < 0 || num_layers > 10) {
		syslog(LOG_ERR, "Number of layers must be between 1 and 10. Obtained %d. Terminating", num_layers);
		exit(1);
	} else {
		syslog(LOG_DEBUG, "Number of Layers found %d", num_layers);
	}
	layers = (struct layer*) calloc(num_layers, sizeof(layer));
	if (layers == NULL) {
		syslog(LOG_ERR, "Unable to allocate memory for layering info collection");
		exit(1);
	} else {
		syslog (LOG_DEBUG, "Allocated memory for layers");
	}
	for (int i=0; i < num_layers; i++) {
		fscanf(fp, "%d ", (int*) &layers[i].type);
		if(layers[i].type == FULLYCONNECTED) {
			struct fcLayer* fc = &layers[i].fc_layer;
			fscanf(fp, "%d %d %d\n", &fc->input_size, &fc->size, (int*)&fc->activation);
			fc->input_size *= batch_size;
			syslog(LOG_DEBUG, "Layer : %d Input size: %d Neurons: %d Activation %d", i, fc->input_size, fc->size, fc->activation);
			get_matrix(&fc->weights, fc->input_size/batch_size, fc->size,1);
			get_matrix(&fc->bias, fc->size, 1, 1);
		} else if (layers[i].type == CONVOLUTION){
			struct convLayer* cl = &layers[i].conv_layer;
			fscanf(fp, "%d %d %d %d %d %d %d\n", &cl->filter_size, &cl->padding, &cl->stride,
												 &cl->maxpool, &cl->num_channels, (int*) &cl->algorithm, (int*)&cl->mode);
			get_matrix(&cl->filter, cl->filter_size, cl->filter_size, 1);
		} else {
			syslog (LOG_ERR, "Unknown Network Block, Terminating the program");
			exit(1);
		}
	}
	fclose(fp);
	syslog(LOG_DEBUG, "Populated Struct Layers");

	if ((status = (int) cudnnCreate(&cudnn)) != (int) CUDNN_STATUS_SUCCESS) {
		syslog(LOG_ERR, "Unable to create CUDA handlers. Terminating the program");
		exit(1);
	}
	syslog(LOG_DEBUG, "Create cudnn handler");

	if ((status = (int) cublasCreate(&cublas)) != (int) CUBLAS_STATUS_SUCCESS) {
		syslog(LOG_ERR, "Unable to create CUDA handlers. Terminating the program");
		exit(1);
	}
	syslog(LOG_DEBUG, "Create cublas handler");

	get_matrix(&input_image, IMAGE_HEIGHT*IMAGE_WIDTH, batch_size, 1);
	fp = fopen("input_image", "w");
	for (int i= 0; i< IMAGE_WIDTH*IMAGE_HEIGHT*batch_size; i++) {
			fprintf(fp, "Var %s Id: %d val: %2.3f \n", "Input", i, input_image[i]);
	}
	fclose(fp);
	status = setup_descriptors (&desc, num_layers, layers);
	if(status != 0) {
		syslog(LOG_ERR, "Error while Descriptor Setup. Terminating the program");
		exit(1);
	}
	syslog(LOG_DEBUG, "Created Descriptors");
	status = configure_descriptors(&cudnn, desc, num_layers, layers, batch_size);
	if(status != 0) {`
			syslog(LOG_ERR, "Error while Descriptor config. Terminating the program");
			exit(1);
	}
	syslog(LOG_DEBUG, "Configured Descriptors");

/*	if (layers[num_layers - 1].type == CONVOLUTION) {
		status = cudnnGetTensor4dDescriptor((desc[num_layers - 1].output_desc), &t, &n, &c, &h, &w, NULL, NULL, NULL, NULL);
		if (status != 0) {
			syslog(LOG_ERR, "Error while determining Output vec size. Terminating the program");
			exit(1);
		}
		create_output_arrays_in_gpu(&h_y, &d_y, &h_one_vector, &d_one_vector, n*c, h*w);
	}
	else if (layers[num_layers - 1].type == FULLYCONNECTED) {
		if ( create_output_arrays_in_gpu(&h_y,&d_y, &h_one_vector,&d_one_vector, batch_size, layers[num_layers - 1].fc_layer.size) != 0) {
			syslog(LOG_ERR, "Unable to create h_y, d_y, h_onevector and d_one_vector");
			exit(1);
		}
	//	printf("\n\n Printing inside Main Function \n\n");
	//	print_matrix(h_y, layers[num_layers - 1].fc_layer.size, batch_size);
	} else {
		syslog(LOG_ERR, "Unknown Network Architecture %d. Terminating the program", layers[num_layers - 1].type);
		exit(1);
	}
*/
	syslog(LOG_DEBUG, "Created yHat");

	allocate_memory(desc, cost, layers, num_layers, batch_size) ;
	if(status != 0) {
		destroy_descriptors(desc, cost, num_layers);
		syslog(LOG_ERR, "Error while allocating Memory. Terminating the program");
		exit(1);
	}
	syslog(LOG_DEBUG, "Allocated Device Memory for Input/Output/Weights");

	copy_input_to_device(desc, layers, num_layers, input_image, batch_size);
	if(status != 0) {
		destroy_descriptors(desc, cost, num_layers);
		syslog(LOG_ERR, "Error while Copying data to Device. Terminating the program");
		exit(1);
	}
	syslog(LOG_DEBUG, "Copied Input to Device");
	float diff_1 = 0;
	float diff_2 = 0;
	struct timeval start_timeval, end_timeval;
	for(int i=0; i < num_turns; i++) {
	gettimeofday(&start_timeval, NULL);
	ff_stat = feedforward(&cudnn, &cublas ,  desc, layers, num_layers, batch_size);
	gettimeofday(&end_timeval, NULL);
	diff_1 += (end_timeval.tv_sec - start_timeval.tv_sec)*1000.0 + (end_timeval.tv_usec - start_timeval.tv_usec)*1.0/1000.0;


	if(ff_stat.failure != NONE) {
		syslog(LOG_ERR, "Error in Feed-forward. Error in %d . Received CUDNN Error %d CUBLAS ERROR %d", ff_stat.layer, ff_stat.cudnn_stat, ff_stat.cublas_stat);
		destroy_descriptors(desc, cost, num_layers);
		exit(1);
	}
	syslog(LOG_DEBUG, "Completed Feedforward");
	gettimeofday(&start_timeval, NULL);
	status = computecost(d_y, desc[num_layers-1].d_output, d_one_vector, layers[num_layers - 1].fc_layer.size, batch_size, cublas, &cost);
	gettimeofday(&end_timeval, NULL);
	diff_2 +=  (end_timeval.tv_sec - start_timeval.tv_sec)*1000.0 + (end_timeval.tv_usec - start_timeval.tv_usec)*1.0/1000.0;
	if(status != 0) {
		destroy_descriptors(desc, cost, num_layers);
		syslog(LOG_ERR, "Error in Compute cost. Terminating the program");
		exit(1);
	}
	syslog(LOG_DEBUG, "Computed the Cost");
	}
	double flop_per_cycle = gigaFlop(layers, num_layers, batch_size)*1.0/1e9;
	printf("The cost of the Cost Function using GPU is %2.7f \n", cost);
	printf("Feedforward Time: %2.5f ms\n",diff_1/num_turns);
	printf("Compute cost Time: %2.5f ms\n",diff_2/num_turns);
	printf("Total Time on GPU: %2.3f ms \n", (diff_1+diff_2)/num_turns);
	printf("GPU Giga Flops: %2.4f\n", (double)flop_per_cycle *num_turns*1000/(diff_1+diff_2));
	syslog(LOG_DEBUG, "Starting CPU based Computation");
	float cost_cpu = 0;
	gettimeofday(&start_timeval, NULL);
	for(int i=0; i< num_turns; i++) {
	NNbyCPU(layers, num_layers, input_image, h_y, batch_size, &cost_cpu);
	}
	gettimeofday(&end_timeval, NULL);
	diff_2 =  (end_timeval.tv_sec - start_timeval.tv_sec)*1000.0 + (end_timeval.tv_usec - start_timeval.tv_usec)*1.0/1000.0;
	printf("The cost of the Optimization Function using CPU is %2.7f \n", cost_cpu);
	printf("Time taken to execute on CPU: %2.3f ms \n", diff_2*1.0/(1.0*num_turns));
	printf("CPU Giga Flops: %2.3f\n", (double)flop_per_cycle*num_turns*1000.0/diff_2);
	printf("Error between CPU and GPU: %2.5f %% \n", (cost-cost_cpu)*100.0/cost);
	status = destroy_descriptors (desc, num_layers);
	if(status != 0) {
		syslog(LOG_ERR, "Descriptors could not be cleaned up. Terminating....");
		exit(1);
	}
	/*
	status = delete_output_arrays_from_gpu(h_y, d_y, h_one_vector, d_one_vector);
	if(status != 0) {
		syslog(LOG_ERR, "Error in Cleanup. Terminating the program");
		exit(1);
	}
	*/
	status = destroy_layers(layers, input_image, num_layers);

	syslog(LOG_DEBUG, "Destroyed Descriptors");
}
