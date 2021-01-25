/**
 * Forward.h
 * 
 * In this header file, we define 
 * the function that feeds forward 
 * a vector to the given neural
 * network.
 */

#pragma once

#include "Common.h"
#include "Neural.h"
#include "Activation.h"

void feedforward(MLP* fcn, long double** (&Z), long double** (&A));
