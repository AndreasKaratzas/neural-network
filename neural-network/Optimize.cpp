
#include "Optimize.h"

/**
 * Computes each neuron's error of a given neural network.
 * 
 * @param[in, out] fcn the neural network instance
 * @param[in] y_train the expected output of the model for a given input
 * @param[in] A the matrix with the neurons' outputs
 * @param[in, out] delta the matrix with all the neurons' errors
 * 
 * @note Although passed by reference, the neural network is not altered.
 * 
 * @note    Using the `REGISTER` variable, we can take advantage of the `reduction()` routine. 
 *          The `delta` container is not contiguous and therefore cannot be 'reduced'.
 * 
 * @note    Although there was no need for the purposes of the project to compute the error of more 
 *          than 1 (one) hidden layers, there is a loop that does exactly that, for completeness.
 */
void backprop(MLP* fcn, long double* y_train, long double** A, long double** (&delta))
{
    int layer, neuron, synapse, dynamic_size;

    long double* REGISTER;                                                                                                                                                  /// Defines a temporary container that enables further optimizations

#pragma omp parallel for simd num_threads(N_THREADS) schedule(runtime) default(none) private(neuron) shared(y_train, fcn, delta, A)
    for (neuron = 0; neuron < fcn->sizes[fcn->num_layers - 1]; neuron += 1)
    {
        delta[fcn->num_layers - 2][neuron] = (A[fcn->num_layers - 1][neuron] - y_train[neuron]) * fcn->autograd(A[fcn->num_layers - 1][neuron]);                            /// Computes the error of the neurons in the last layer
    }

    dynamic_size = fcn->sizes[fcn->num_layers - 2];                                                                                                                         /// Declares ammount of memory to store `delta` corresponding to the neurons of the last *hidden* layer

    while (NULL == (REGISTER = (long double*)malloc(dynamic_size * sizeof(long double)))) {}                                                                                /// Allocates the ammount of memory computed above
    
    std::fill_n(REGISTER, dynamic_size, 0.0);                                                                                                                               /// Initializes the allocated container

#pragma omp parallel for collapse(2) reduction(+ : REGISTER[0:dynamic_size]) num_threads(N_THREADS) schedule(runtime) default(none) private(synapse, neuron) shared(REGISTER, fcn, delta)
    for (synapse = 0; synapse < fcn->sizes[fcn->num_layers - 2]; synapse += 1)
    {
        for (neuron = 0; neuron < fcn->sizes[fcn->num_layers - 1]; neuron += 1)
        {
            REGISTER[synapse] += fcn->weights[fcn->num_layers - 2][neuron][synapse] * delta[fcn->num_layers - 2][neuron];                                                   /// Computes the first factor of the error of neurons for the last *hidden* layer
        }
    }

    memmove(delta[fcn->num_layers - 3], REGISTER, dynamic_size * sizeof(long double));                                                                                      /// Moves the computed results from the temporary container to `delta`

    free(REGISTER);                                                                                                                                                         /// Deallocated memory space requested for the `REGISTER` container

#pragma omp parallel for simd num_threads(N_THREADS) schedule(runtime) default(none) private(synapse) shared(fcn, delta, A)
    for (synapse = 0; synapse < fcn->sizes[fcn->num_layers - 2]; synapse += 1)
    {
        delta[fcn->num_layers - 3][synapse] = delta[fcn->num_layers - 3][synapse] * fcn->autograd(A[fcn->num_layers - 2][synapse]);                                         /// Computes the total neuron error for each neuron in the last *hidden* layer
    }

    for (layer = 2; layer < fcn->num_layers - 1; layer += 1)                                                                                                                /// Computes the error for neurons in the remaining hidden layers using the same method
    {
        dynamic_size = fcn->sizes[fcn->num_layers - layer - 1];

        while (NULL == (REGISTER = (long double*)malloc(dynamic_size * sizeof(long double)))) {}

        std::fill_n(REGISTER, dynamic_size, 0.0);

#pragma omp parallel for collapse(2) reduction(+ : REGISTER[0:dynamic_size]) num_threads(N_THREADS) schedule(runtime) default(none) private(synapse, neuron) shared(layer, REGISTER, fcn, delta, A)
        for (synapse = 0; synapse < fcn->sizes[fcn->num_layers - layer - 1]; synapse += 1)
        {
            for (neuron = 0; neuron < fcn->sizes[fcn->num_layers - layer] - 1; neuron += 1)                                                                                 /// There is no synapse between the bias at layer `l` and any neuron at layer `l - 1`
            {
                REGISTER[synapse] += fcn->weights[fcn->num_layers - layer - 1][neuron][synapse] * delta[fcn->num_layers - layer - 1][neuron];
            }
        }

        memmove(delta[fcn->num_layers - layer - 2], REGISTER, dynamic_size * sizeof(long double));

        free(REGISTER);

#pragma omp parallel for simd num_threads(N_THREADS) schedule(runtime) default(none) private(synapse) shared(fcn, delta, A, layer)
        for (synapse = 0; synapse < fcn->sizes[fcn->num_layers - layer - 1]; synapse += 1)
        {
            delta[fcn->num_layers - layer - 2][synapse] = delta[fcn->num_layers - layer - 2][synapse] * fcn->autograd(A[fcn->num_layers - layer - 1][synapse]);
        }
    }
}

/**
 * Optimizes weights by subtracted the precomputed error corresponding to each neuron pair (synapse).
 * 
 * @param[in, out] fcn the neural network instance
 * @param[in] delta the matrix with all the neurons' errors
 * @param[in] A the matrix with the neurons' outputs
 */
void optimize(MLP* fcn, long double** delta, long double **A)
{
    int layer, neuron, synapse;

#pragma omp parallel for collapse(2) schedule(runtime) num_threads(N_THREADS) default(none) private(neuron, synapse) shared(fcn, LEARNING_RATE, A, delta)
    for (neuron = 0; neuron < fcn->sizes[fcn->num_layers - 1]; neuron += 1)                                                                                                         /// Loops through all neurons in the last layer
    {
        for (synapse = 0; synapse < fcn->sizes[fcn->num_layers - 2]; synapse += 1)                                                                                                  /// Loops through all neurons in the last *hidden* layer
        {
            fcn->weights[fcn->num_layers - 2][neuron][synapse] -= LEARNING_RATE * delta[fcn->num_layers - 2][neuron] * A[fcn->num_layers - 2][synapse];                             /// Optimizes weights between those synapses
        }
    }
    
    for (layer = 2; layer < fcn->num_layers; layer += 1)                                                                                                                            /// Loops through all the other layers
    {
#pragma omp parallel for collapse(2) schedule(runtime) num_threads(N_THREADS) default(none) private(neuron, synapse) shared(fcn, LEARNING_RATE, A, delta, layer)
        for (neuron = 0; neuron < fcn->sizes[fcn->num_layers - layer] - 1; neuron += 1)
        {
            for (synapse = 0; synapse < fcn->sizes[fcn->num_layers - layer - 1]; synapse += 1)
            {
                fcn->weights[fcn->num_layers - layer - 1][neuron][synapse] -= LEARNING_RATE * delta[fcn->num_layers - layer - 1][neuron] * A[fcn->num_layers - layer - 1][synapse]; /// Uses the same method to optimize the rest of the model's synapses
            }
        }
    }
}
