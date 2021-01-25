/**
 * Loss.h
 * 
 * In this header file, we define the 
 * `Mean Squared Error` Loss function
 * which is used in this project to
 * estimate the model's loss.
 */

#pragma once

#include "Common.h"

long double mse_loss(long double* y_pred, long double* y_real, int out_dim);
