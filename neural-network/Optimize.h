/**
 * Optimize.h
 * 
 * In this header file, we define all 
 * the functions used for the optimization
 * of the neural network. The model is 
 * optimized using the back propagation
 * optimization algorithm. The `backprop`
 * function computes the error of a neuron.
 * The `optimize` function applies that error
 * to each synapse optimizing the weights of
 * those synapses.
 */
#pragma once

#include "Activation.h"
#include "Common.h"
#include "Neural.h"
#include "Utility.h"

void backprop(MLP* fcn, long double* y_train, long double** A, long double** (&delta));
void optimize(MLP* fcn, long double** delta, long double** A);
