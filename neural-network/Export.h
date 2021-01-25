/**
 * Export.h
 * 
 * In this header file, we define a
 * function that exports a long double
 * precision floating point vector into 
 * a CSV file.
 */

#pragma once

#include "Common.h"

void export_double_vector(long double* (&container), char* filename, int dim);
