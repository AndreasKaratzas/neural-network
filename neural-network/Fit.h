/**
 * Fit.h
 * 
 * In this header file, we define
 * all the routines necessary for 
 * the model's fitting. There is a
 * function for the model's training
 * and a function for the model's 
 * evaluation.
 */

#pragma once

#include "Loss.h"
#include "Common.h"
#include "Neural.h"
#include "Export.h"
#include "Forward.h"
#include "Utility.h"
#include "Optimize.h"
#include "Accuracy.h"
#include "Interface.h"

void train(MLP* fcn, long double** (&x_train), long double** y_train, long double* (&train_loss), int* (&train_accuracy), int* (&train_dim));
long double test(MLP* fcn, long double** (&x_test), long double** (&y_test), int* (&test_dim));
