/**
 * Neural.h
 * 
 * In this header file, we define all 
 * the functions that handle factors of
 * the neural network. Those factors 
 * include the weights of the network as
 * well as the activation function used
 * acrross the model.
 */

#pragma once

#include "Common.h"

typedef struct MLP                                                  /// Structure of the neural network
{
    int num_layers;                                                 /// A neural network consists of a number of layers
    int* sizes;                                                     /// The (different) sizes of those layers
    long double*** weights;                                         /// The weights of the synapses between the neurons in the network

    long double (*(filter))(long double x);                         /// The common activation function of the neurons
    long double (*(autograd))(long double x);                       /// The gradient of the activation function

} MLP;

long double rand_probability(long double min, long double max);
void set_num_layers(MLP* nn, int L);
void set_filter(MLP* nn, long double (*(activation))(long double x));
void set_autograd(MLP* nn, long double (*(derivative))(long double x));
void set_sizes(MLP* nn, int layer_count, int* args);
void set_weights(MLP* nn, int layer_count, int* args);
void delete_weights(MLP* nn);
void compile_nn(MLP* nn, int* args, int layer_count, long double (*(activation))(long double x), long double (*(derivative))(long double x));
void delete_nn(MLP* nn);
