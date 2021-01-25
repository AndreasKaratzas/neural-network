/**
 * Preprocess.h 
 * 
 * In this header file, we define all 
 * functions that load the given dataset
 * to the executable. The defined routines
 * split the dataset into input examples and
 * the corresponding (expected) outputs. 
 */

#pragma once

#include "Common.h"
#include "Progressbar.h"

ssize_t getline(char** lineptr, size_t* n, FILE* stream);
void read_csv(const char* filename, long double** (&X), long double** (&Y), int* (&dimensions), int dataset_flag);
void train_test_split(long double** (&x_train), long double** (&y_train), int* (&train_dim), long double** (&x_test), long double** (&y_test), int* (&test_dim));
