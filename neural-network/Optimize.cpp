
#include "Neural.hpp"

/**
 * Computes each neuron's error of a given neural network.

 * @param[in, out] Y the expected output of the model for a given input
 * 
 * @note Although passed by reference, the `Y` placeholder is not altered.
 *
 * @note    Using the `REGISTER` variable, we can take advantage of the `reduction()` routine.
 *          The `delta` container is not contiguous and therefore cannot be 'reduced'.
 *
 * @note    Although there was no need for the purposes of the project to compute the error of more
 *          than 1 (one) hidden layers, there is a loop that does exactly that, for completeness.
 */
void nn::back_propagation(double* (&Y))
{
#pragma omp parallel for num_threads(N_THREADS)
    for (int neuron = 0; neuron < layers[layers.size() - 1]; neuron += 1)
    {
        delta[layers.size() - 2][neuron] = (a[layers.size() - 1][neuron] - Y[neuron]) * sig_derivative(a[layers.size() - 1][neuron]);               /// Computes the error of the neurons in the last layer
    }
#pragma omp parallel for num_threads(N_THREADS)
    for (int synapse = 0; synapse < layers[layers.size() - 2]; synapse += 1)
    {
        double REGISTER = 0.0;
#pragma omp simd reduction(+ : REGISTER)
        for (int neuron = 0; neuron < layers[layers.size() - 1]; neuron += 1)
        {
            REGISTER += weights[layers.size() - 2][neuron][synapse] * delta[layers.size() - 2][neuron];                                    /// Computes the first factor of the error of neurons for the last *hidden* layer
        }
        delta[layers.size() - 3][synapse] = REGISTER;
    }

#pragma omp parallel for num_threads(N_THREADS)
    for (int synapse = 0; synapse < layers[layers.size() - 2]; synapse += 1)
    {
        delta[layers.size() - 3][synapse] = delta[layers.size() - 3][synapse] * sig_derivative(a[layers.size() - 2][synapse]);                      /// Computes the total neuron error for each neuron in the last *hidden* layer
    }

    for (int layer = 2; layer < layers.size() - 1; layer += 1)                                                                                      /// Computes the error for neurons in the remaining hidden layers using the same method
    {
#pragma omp parallel for num_threads(N_THREADS)
        for (int synapse = 0; synapse < layers[layers.size() - layer - 1]; synapse += 1)
        {
            double REGISTER = 0.0;
#pragma omp simd reduction(+ : REGISTER)
            for (int neuron = 0; neuron < layers[layers.size() - layer] - 1; neuron += 1)                                                           /// There is no synapse between the bias at layer `l` and any neuron at layer `l - 1`
            {
                REGISTER += weights[layers.size() - layer - 1][neuron][synapse] * delta[layers.size() - layer - 1][neuron];
            }
            delta[layers.size() - layer - 2][synapse] = REGISTER;
        }

#pragma omp parallel for num_threads(N_THREADS)
        for (int synapse = 0; synapse < layers[layers.size() - layer - 1]; synapse += 1)
        {
            delta[layers.size() - layer - 2][synapse] = delta[layers.size() - layer - 2][synapse] * sig_derivative(a[layers.size() - layer - 1][synapse]);
        }                                                                                                                          /// Deallocates memory space requested for the `REGISTER` container
    }
}

/**
 * Optimizes weights by subtracting the precomputed error corresponding to each neuron pair (synapse).
 */
void nn::optimize(void)
{
#pragma omp parallel for num_threads(N_THREADS)
    for (int neuron = 0; neuron < layers[layers.size() - 1]; neuron += 1)                                                                           /// Loops through all neurons in the last layer
    {
#pragma omp simd
        for (int synapse = 0; synapse < layers[layers.size() - 2]; synapse += 1)                                                                    /// Loops through all neurons in the last *hidden* layer
        {
            weights[layers.size() - 2][neuron][synapse] -= LEARNING_RATE * delta[layers.size() - 2][neuron] * a[layers.size() - 2][synapse];        /// Optimizes weights between those synapses
        }
    }

    for (int layer = 2; layer < layers.size(); layer += 1)                                                                                          /// Loops through all the other layers
    {
#pragma omp parallel for num_threads(N_THREADS)
        for (int neuron = 0; neuron < layers[layers.size() - layer] - 1; neuron += 1)
        {
#pragma omp simd
            for (int synapse = 0; synapse < layers[layers.size() - layer - 1]; synapse += 1)                                                        /// Uses the same method to optimize the rest of the model's synapses
            {
                weights[layers.size() - layer - 1][neuron][synapse] -= LEARNING_RATE * delta[layers.size() - layer - 1][neuron] * a[layers.size() - layer - 1][synapse];
            }
        }
    }
}
