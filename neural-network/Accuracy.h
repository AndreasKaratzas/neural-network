/**
 * Accuracy.h
 * 
 * In this header file, we define a function
 * that computes the accuracy of the given 
 * neural network.
 */

#pragma once

#include "Common.h"

int model_accuracy(long double *y_pred, long double *y_target, int dim);
