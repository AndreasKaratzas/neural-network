
#include "Neural.hpp"

/**
 * Feeds forward the given model a given input vector.
 *
 * @note    To exploit the full capabilities of the OpenMP framework, we use `collapse()` routine wherever possible.
 *          To use this routine, the given vector must be contiguous, therefore a 2D dynamic array cannot be collapsed.
 *          That's why there are temporary variables called `REGISTERS`, which are the 1D temporary image of those 2D
 *          vectors. Those registers are used during the parallel computations, and then we utilize the `memmove()`
 *          routine which has O(1) time complexity and transfers the computations back to the 2D vectors.
 */
void nn::forward(void)
{
    int dynamic_size;
    double* REGISTER;

    for (int layer = 1; layer < layers.size() - 1; layer += 1)
    {
        dynamic_size = layers[layer] - 1;                                                                       /// fetches the number of neurons of the currently parsed hidden layer excluding the layer's bias

        REGISTER = (double*)calloc(dynamic_size, sizeof(double));                                               /// Allocates temporary memory

        if (REGISTER == NULL)
        {
            perror("calloc() failed");                                                                          /// Masks memory allocation fault
        }

#pragma omp parallel for collapse(2) reduction(+ : REGISTER[0: dynamic_size]) num_threads(N_THREADS) schedule(runtime)
        for (int neuron = 0; neuron < layers[layer] - 1; neuron += 1)                                           /// Iterates through the hidden layer's neurons
        {
            for (int synapse = 0; synapse < layers[layer - 1]; synapse += 1)                                    /// Iterates throught the previous layer
            {
                REGISTER[neuron] += weights[layer - 1][neuron][synapse] * a[layer - 1][synapse];                /// Implements forward propagation for all hidden layers
            }
        }

        memmove(z[layer], REGISTER, dynamic_size * sizeof(double));                                             /// Moves the results to the main data container

#pragma omp parallel for simd num_threads(N_THREADS) schedule(runtime)
        for (int neuron = 0; neuron < layers[layer] - 1; neuron += 1)
        {
            a[layer][neuron] = sigmoid(z[layer][neuron]);                                                       /// Applies model's ativation function to computed results
        }

        free(REGISTER);                                                                                         /// Deallocates the temporary container off the memory
    }

    dynamic_size = layers[layers.size() - 1];                                                                   /// Holds the number of neurons of the output layer

    REGISTER = (double*)calloc(dynamic_size, sizeof(double));                                                   /// Allocates temporary memory

    if (REGISTER == NULL)
    {
        perror("calloc() failed");                                                                              /// Masks memory allocation fault
    }

#pragma omp parallel for collapse(2) reduction(+ : REGISTER[0: dynamic_size]) num_threads(N_THREADS) schedule(runtime)
    for (int neuron = 0; neuron < layers[layers.size() - 1]; neuron += 1)
    {
        for (int synapse = 0; synapse < layers[layers.size() - 2]; synapse += 1)
        {
            REGISTER[neuron] += weights[layers.size() - 2][neuron][synapse] * a[layers.size() - 2][synapse];    /// Implements forward propagation for the output layer
        }
    }

    memmove(z[layers.size() - 1], REGISTER, dynamic_size * sizeof(double));

#pragma omp parallel for simd num_threads(N_THREADS) schedule(runtime)
    for (int neuron = 0; neuron < layers[layers.size() - 1]; neuron += 1)
    {
        a[layers.size() - 1][neuron] = sigmoid(z[layers.size() - 1][neuron]);                                   /// Applies model's ativation function to computed results
    }

    free(REGISTER);                                                                                             /// Deallocates the temporary container off the memory
}
