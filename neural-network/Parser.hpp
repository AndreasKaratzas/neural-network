/**
 * Parser.hpp
 *
 * In this header file, we define all
 * the functions that help parse the user's
 * arguments. Those arguments include the
 * sizes of the different layers to define
 * a neural network. The user can also choose
 * to review the programs usage.
 */

#pragma once

#include "Interface.hpp"

int parse_integer(char* argv);
void parse_arguments(int argc, char* argv[], std::vector<int>& vec);
