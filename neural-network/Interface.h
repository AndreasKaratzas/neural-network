/**
 * Interface.h
 * 
 * In this header file, we implement a
 * basic User Interface using the Command
 * Line Window. The defined functions are
 * used to print information about the data
 * processed by the neural network, about
 * the dataset, about the neural network's
 * progress and help the user understand
 * how to use the project.
 */

#pragma once

#include "Common.h"
#include "Utility.h"

void usage(char* filename);
void print_2D_DOUBLE_vector(long double** container, int num_layers, int* sizes, char* container_typename);
void print_3D_DOUBLE_vector(long double*** container, int num_layers, int* sizes, char* container_typename);
void print_2D_INT_array(int** container, char* container_typename, int dim_1, int dim_2);
void print_1D_INT_array(int* container, char* container_typename, int dim);
void print_parser_results(int input_size, int* hidden_size, int hidden_dim, int output_size, char* activation);
void print_epoch_stats(int epoch, long double epoch_loss, int epoch_accuracy, double benchmark);
void print_delta(long double** delta, int num_layers, int* sizes, char* container_typename);
