/**
 * Preprocess.h
 *
 * In this header file, we define all
 * functions that load the given dataset
 * to the executable. The defined routines
 * split the dataset into input examples and
 * the corresponding (expected) outputs.
 */

#pragma once

#include "Interface.hpp"

class dataset
{
public:
    int samples, dimensions, classes;
    double** X, ** Y;

    ssize_t getline(char** lineptr, size_t* n, FILE* stream);
    void read_csv(const char* filename, int dataset_flag);
    int get_label(int sample);
    void print_dataset();

    dataset()
    {
        classes = MNIST_CLASSES;
    }

    ~dataset()
    {
        for (int i = 0; i < samples; i += 1)
        {
            delete[] X[i];
            delete[] Y[i];
        }
        delete[] X;
        delete[] Y;
    }
};
