
#include "Forward.h"

/**
 * Feeds forward the given model a given input vector.
 * 
 * @param[in] fcn the `Fully Connected Network`
 * @param[in] Z the neurons' unfiltered data (before activation function). This correspinds to `DL[]` container
 * @param[in] A the neurons' filtered data (after activation function). This correspinds to `OL[]` container
 * 
 * @note    To exploit the full capabilities of the OpenMP framework, we use `collapse()` routine wherever possible. 
 *          To use this routine, the given vector must be contiguous, therefore a 2D dynamic array cannot be collapsed.
 *          That's why there are temporary variables called `REGISTERS`, which are the 1D temporary image of those 2D 
 *          vectors. Those registers are used during the parallel computations, and then we utilize the `memmove()`
 *          routine which has O(1) time complexity and transfers the computations back to the 2D vectors.
 */
void feedforward(MLP* fcn, long double** (&Z), long double** (&A))
{
    int layer, neuron, synapse, dynamic_size;

    long double *REGISTER_H, *REGISTER_O;

    for (layer = 1; layer < fcn->num_layers - 1; layer += 1)
    {
        dynamic_size = fcn->sizes[layer] - 1;                                                                               /// fetches the number of neurons of the currently parsed hidden layer excluding the layer's bias

        while (NULL == (REGISTER_H = (long double*)malloc(dynamic_size * sizeof(long double)))) { }                         /// `REGISTER_H` refers to the register of the currently parsed hidden layer

        std::fill_n(REGISTER_H, dynamic_size, 0.0);                                                                         /// Initializes `REGISTER_H` with zeros

#pragma omp parallel for collapse(2) reduction(+ : REGISTER_H[0:dynamic_size]) num_threads(N_THREADS) schedule(runtime) default(none) private(neuron, synapse) shared(layer, REGISTER_H, fcn, A)
        for (neuron = 0; neuron < fcn->sizes[layer] - 1; neuron += 1)                                                       /// Iterates through the hidden layer's neurons
        {
            for (synapse = 0; synapse < fcn->sizes[layer - 1]; synapse += 1)                                                /// Iterates throught the previous layer
            {
                REGISTER_H[neuron] += fcn->weights[layer - 1][neuron][synapse] * A[layer - 1][synapse];                     /// Implements forward propagation for all hidden layers
            }
        }

        memmove(Z[layer], REGISTER_H, dynamic_size * sizeof(long double));                                                  /// Moves the results to the main data container

#pragma omp parallel for simd num_threads(N_THREADS) schedule(runtime) default(none) private(neuron) shared(fcn, Z, A, layer)
        for (neuron = 0; neuron < fcn->sizes[layer] - 1; neuron += 1)
        {
            A[layer][neuron] = fcn->filter(Z[layer][neuron]);                                                               /// Applies model's ativation function to computed results
        }

        free(REGISTER_H);                                                                                                   /// Deallocates the temporary container off the memory
    }

    dynamic_size = fcn->sizes[fcn->num_layers - 1];                                                                         /// Holds the number of neurons of the output layer

    while (NULL == (REGISTER_O = (long double*)malloc(dynamic_size * sizeof(long double)))) {}                              /// `REGISTER_O` refers to the register of the output layer

    std::fill_n(REGISTER_O, dynamic_size, 0.0);                                                                             /// Initializes `REGISTER_O` with zeros

#pragma omp parallel for collapse(2) reduction(+ : REGISTER_O[0:dynamic_size]) num_threads(N_THREADS) schedule(runtime) default(none) private(neuron, synapse) shared(REGISTER_O, fcn, A)
    for (neuron = 0; neuron < fcn->sizes[fcn->num_layers - 1]; neuron += 1)
    {
        for (synapse = 0; synapse < fcn->sizes[fcn->num_layers - 2]; synapse += 1)
        {
            REGISTER_O[neuron] += fcn->weights[fcn->num_layers - 2][neuron][synapse] * A[fcn->num_layers - 2][synapse];     /// Implements forward propagation for the output layer
        }
    }

    memmove(Z[fcn->num_layers - 1], REGISTER_O, dynamic_size * sizeof(long double));                                        /// Moves the results to the main data container

#pragma omp parallel for simd num_threads(N_THREADS) schedule(runtime) default(none) private(neuron) shared(A, fcn, Z)
    for (neuron = 0; neuron < fcn->sizes[fcn->num_layers - 1]; neuron += 1)
    {
        A[fcn->num_layers - 1][neuron] = fcn->filter(Z[fcn->num_layers - 1][neuron]);                                       /// Applies model's ativation function to computed results
    }

    free(REGISTER_O);                                                                                                       /// Deallocates the temporary container off the memory
}
