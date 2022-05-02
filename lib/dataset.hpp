/**
 * dataset.hpp
 *
 * In this header file, we define a
 * class that handles a dataset. The class
 * has a function that imports the dataset
 * directly from a CSV file. There is also
 * a function that prints out the input and
 * expected output of a dataset instance.
 * Finally, while parsing the CSV file, the
 * data is split into input examples and
 * the corresponding (expected) outputs.
 */

#pragma once

#include "interface.hpp"

 /**
  * Implementation of a dataset class.
  *
  * Upon a dataset creation, the developer
  * has to call `read_csv` providing the
  * requested arguments to start using
  * the created dataset instance. The dataset
  * has got an attribute (variable) `classes`
  * which is initialized after the number of
  * classes in a dataset. For the project's
  * purposes, this attribute has been initialized
  * with 10 (ten), since there are 10 classes in the
  * MNIST fashion dataset.
  */
class dataset
{
public:
    int samples, dimensions, classes;
    double** X, ** Y;

    ssize_t getline(char** lineptr, size_t* n, FILE* stream);
    void read_csv(const char* filename, int dataset_flag, double x_max);
    int get_label(int sample);
    void print_dataset(void);

    dataset(int classes, int samples) :
        classes{ classes },
        samples{ samples }
    {
        dimensions = 0;
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
