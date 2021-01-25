/**
 * Activation.h
 * 
 * In this header file, we define the  
 * activation functions which are used 
 * in the neural network to filter the
 * neurons' value. We also define the 
 * derivatives of those functions used
 * during back propagation.
 */

#pragma once

#include "Common.h"

long double sigmoid(long double x);
long double sig_derivative(long double x);
long double relu(long double x);
long double rel_derivative(long double x);
