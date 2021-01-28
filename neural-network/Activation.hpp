/**
 * Activation.hpp
 * 
 * In this header file, we define
 * all neuron activtion functions.
 * Specifically, there is an 
 * implementation of the sigmoid
 * and the ReLU activation function.
 * There are also the corresponding 
 * derivatives of those functions
 * to be used during neuron error
 * computation (back propagation 
 * algorithm).
 */

#pragma once

#include "Common.hpp"

double exp(double x);
double sigmoid(double x);
double fast_sigmoid(double x);
double sig_derivative(double x);
double relu(double x);
double rel_derivative(double x);
