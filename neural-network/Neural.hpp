/**
 * Neural.hpp
 *
 * In this header file, we define 
 * a template for the custom neural 
 * network.
 */

#pragma once

#include "Common.hpp"
#include "Dataset.hpp"
#include "Activation.hpp"

/**
 * Implements a Multi Layer Perceptron model.
 * 
 * The neural network works with sigmoid 
 * activation function and Mean Squared 
 * Error loss function. However, there 
 * is potential for support of various 
 * activation and loss functions. The
 * performance of the neural network 
 * class has been optimized with OpenMP
 * framework.
 */
class nn
{
public:
    double** z, ** a, ** delta, *** weights;

    std::vector<int> layers;

    void set_layers(const std::vector<int>& l);
    void set_z(const std::vector<int>& l);
    void set_a(const std::vector<int>& l);
    void set_delta(const std::vector<int>& l);
    void set_weights(const std::vector<int>& l, const double min, const double max);
    void compile(const std::vector<int>& l, const double min, const double max);
    void zero_grad(double* (&X));
    void forward(void);
    void back_propagation(double* (&Y));
    void optimize(void);
    int get_label(double* (&y_pred));
    int predict(double* (&X));
    double mse_loss(double* (&Y), int dim);
    int accuracy(double* (&Y), int dim);
    void fit(dataset(&TRAIN));
    void evaluate(dataset(&TEST));
    void export_weights(std::string filename);
    void summary(void);

    nn()
    {

    }

    ~nn()
    {
        for (int i = 0; i < layers.size(); i += 1)
        {
            delete[] z[i];
            delete[] a[i];
        }
        delete[] z;
        delete[] a;
        for (int i = 0; i < layers.size() - 1; i += 1)
        {
            delete[] delta[i];
        }
        delete[] delta;
        for (int i = 1; i < layers.size(); i += 1)
        {
            for (int j = 0; j < layers[i]; j += 1)
            {
                delete[] weights[i][j];
            }
            delete[] weights[i];
        }
        delete[] weights;

        layers.clear();
        layers.shrink_to_fit();
    }
};
