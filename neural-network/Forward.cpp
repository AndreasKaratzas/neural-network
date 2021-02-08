
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
    for (int layer = 1; layer < layers.size() - 1; layer += 1)
    {
#pragma omp parallel for num_threads(N_THREADS)
        for (int neuron = 0; neuron < layers[layer] - 1; neuron += 1)                                           /// Iterates through the hidden layer's neurons
        {
            double REGISTER = 0.0;
#pragma omp simd reduction(+ : REGISTER)
            for (int synapse = 0; synapse < layers[layer - 1]; synapse += 1)                                    /// Iterates throught the previous layer
            {
                REGISTER += weights[layer - 1][neuron][synapse] * a[layer - 1][synapse];                        /// Implements forward propagation for all hidden layers
            }
            z[layer][neuron] = REGISTER;
        }

#pragma omp parallel for num_threads(N_THREADS)
        for (int neuron = 0; neuron < layers[layer] - 1; neuron += 1)
        {
            a[layer][neuron] = sigmoid(z[layer][neuron]);                                                       /// Applies model's activation function to computed results
        }
    }

#pragma omp parallel for num_threads(N_THREADS)
    for (int neuron = 0; neuron < layers[layers.size() - 1]; neuron += 1)
    {
        double REGISTER = 0.0;
#pragma omp simd reduction(+ : REGISTER)
        for (int synapse = 0; synapse < layers[layers.size() - 2]; synapse += 1)
        {
            REGISTER += weights[layers.size() - 2][neuron][synapse] * a[layers.size() - 2][synapse];            /// Implements forward propagation for the output layer
        }
        z[layers.size() - 1][neuron] = REGISTER;
    }

#pragma omp parallel for num_threads(N_THREADS)
    for (int neuron = 0; neuron < layers[layers.size() - 1]; neuron += 1)
    {
        a[layers.size() - 1][neuron] = sigmoid(z[layers.size() - 1][neuron]);                                   /// Applies model's activation function to computed results
    }                                                                                                           /// Deallocates the temporary container off the memory
}
