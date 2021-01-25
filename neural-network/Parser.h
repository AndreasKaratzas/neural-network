/**
 * Parser.h
 * 
 * In this header file, we define all
 * the functions that help parse the user's 
 * arguments. Those arguments include the 
 * sizes of the different layers to define
 * a neural network and the activation function
 * used to filter the value of each neuron. The
 * user can also choose to review the programs 
 * usage.
 */

#pragma once

#include "Common.h"
#include "Interface.h"

int parse_integer(char* argv);
int parse_arguments(int argc, char* argv[], int* input_size, int* hidden_size, int* output_size, char* activation);
