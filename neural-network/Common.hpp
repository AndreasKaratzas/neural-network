
/**
 * Common.hpp
 *
 * In this header file, we define the constants
 * used throughout the project. We also
 * include all the header files necessary to
 * make the implementation work.
 */

#pragma once

#include <cmath>                            /// ceil()
#include <array>                            /// std::array
#include <cerrno>                           /// EINVAL
#include <random>                           /// std::random
#include <vector>                           /// std::vector
#include <limits>                           /// std::numeric_limits
#include <cstdio>                           /// printf()
#include <cstring>                          /// memmove()
#include <iomanip>                          /// std::setw
#include <fstream>                          /// std::ostream
#include <cassert>                          /// assert()
#include <cstdlib>                          /// system()
#include <iostream>                         /// std::cout
#include <stdexcept>                        /// std::runtime_error
#include <algorithm>                        /// std::fill_n
#include <inttypes.h>                       /// intptr_t

#include <omp.h>                            /// OpenMP Multiprocessing Programming Framework

#define array_sizeof(type) ((char *)(&type+1)-(char*)(&type))
                                            /// Macro that computes the size of an array
typedef intptr_t ssize_t;                   /// Declares `ssize_t` type that is used in `Preprocessing.h`

constexpr int EPOCHS = 100;                 /// Declares the number of epochs for the model's training
constexpr int N_THREADS = 12;               /// Specifies the number of threads to request from the OS
constexpr int N_ACTIVATIONS = 2;            /// Declares the number of neuron activation functions declared in the project
constexpr int CLI_WINDOW_WIDTH = 50;        /// Defines the length of the progress bar for the project's CLI
constexpr int MNIST_CLASSES = 10;           /// Declares the number of classes found in the MNIST dataset
constexpr double LEARNING_RATE = 0.1;       /// Defines the learning rate for the neural network
constexpr double MNIST_TRAIN = 60000.0;     /// Declares the number of training examples found in the MNIST dataset
constexpr double MNIST_TEST = 10000.0;      /// Declares the number of evaluation examples found in the MNIST dataset
constexpr double EXP = 2.718282;            /// Defines the exponential constant `e`
constexpr double MNIST_MAX_VAL = 255.0;     /// Defines max value found in the input subset of theMNIST dataset
constexpr char TRAINING_DATA_FILEPATH[] = "fashion-mnist_train.csv";
                                            /// Declares the filepath of the MNIST training CSV file
constexpr char EVALUATION_DATA_FILEPATH[] = "fashion-mnist_test.csv";
                                            /// Declares the filepath of the MNIST evaluation CSV file
