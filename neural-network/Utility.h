/**
 * Utility.h
 * 
 * In this header file, we define 
 * utility functions, such as memory 
 * allocation modules and random
 * generators. There is also a decoder
 * function for the neural network's
 * activation function. The driver to 
 * parse the user arguments is also 
 * declared in this header file.
 */

#pragma once

#include "Common.h"
#include "Neural.h"

long double rand_probability(long double min, long double max);
int get_nn_activation(char* activation);
void get_args(int* (&args), int input_layer, int* hidden_layer, int output_layer, int hid_dim);
void allocate_delta(long double** (&delta), MLP* fcn);
void allocate_Z(long double** (&Z), MLP* fcn, long double* x_train);
void allocate_A(long double** (&A), MLP* fcn, long double* x_train);
void zero_grad(MLP* fcn, long double** (&Z), long double** (&A), long double* x_train);
void zero_grad(long double** (&delta), MLP* fcn, long double** (&Z), long double** (&A), long double* x_train);
void deallocate_Z(long double** (&Z), MLP* fcn);
void deallocate_A(long double** (&A), MLP* fcn);
void deallocate_delta(long double** (&delta), MLP* fcn);
void deallocate_dataset(long double** (&x_train), long double** (&y_train), long double** (&x_test), long double** (&y_test), int* train_dim, int* test_dim);
