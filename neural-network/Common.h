/**
 * Common.h
 *
 * In this header file, we define the constants
 * used throughout the project. We also
 * include all the header files necessary to
 * make the implementation work.
 */

#pragma once

#include <time.h>                           /// time()
#include <math.h>                           /// ceil()
#include <stdio.h>                          /// printf()
#include <errno.h>                          /// EINVAL
#include <stdlib.h>                         /// malloc()
#include <string.h>                         /// memmove()
#include <assert.h>                         /// assert()
#include <inttypes.h>                       /// intptr_t

#include <algorithm>                        /// std::fill_n()

#include <omp.h>                            /// OpenMP Multiprocessing Programming Framework

#define array_sizeof(type) ((char *)(&type+1)-(char*)(&type))
                                            /// Macro that computes the size of an array
typedef intptr_t ssize_t;                   /// Declares `ssize_t` type that is used in `Preprocessing.h`

constexpr int EPOCHS = 100;                 /// Declares the number of epochs for the model's training
constexpr int N_THREADS = 12;               /// Specifies the number of threads to request from the OS
constexpr int N_ACTIVATIONS = 2;            /// Declares the number of neuron activation functions declared in the project
constexpr int CLI_WINDOW_WIDTH = 50;        /// Defines the length of the progress bar for the project's CLI
constexpr int MNIST_CLASSES = 10;           /// Declares the number of classes found in the MNIST dataset
constexpr long double LEARNING_RATE = 0.8;  /// Defines the learning rate for the neural network
constexpr long double MNIST_TRAIN = 60000.0;/// Declares the number of training examples found in the MNIST dataset
constexpr long double MNIST_TEST = 10000.0; /// Declares the number of evaluation examples found in the MNIST dataset

constexpr char TRAINING_DATA_FILEPATH[] = "C:/Users/andreas/Documents/workspace/source/repos/neural-network/neural-network/data/fashion-mnist_train.csv";
                                            /// Declares the filepath of the MNIST training CSV file
constexpr char EVALUATION_DATA_FILEPATH[] = "C:/Users/andreas/Documents/workspace/source/repos/neural-network/neural-network/data/fashion-mnist_test.csv";
                                            /// Declares the filepath of the MNIST evaluation CSV file
