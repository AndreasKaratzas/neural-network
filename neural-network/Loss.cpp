
#include "Neural.hpp"

/**
 * Computes the model's MSE loss.
 *
 * @param[in] Y the expected output. This is the ground truth given the same input
 * @param[in] dim the size of the output layer and therefore the size of the `Y` placeholder
 * 
 * @return the total loss based on the model's predictions on a given sample and the corresponding (expected) output
 *
 * @note The given vectors must be of the same dimensions.
 * 
 * @note Although passed by reference, the `Y` placeholder is not altered.
 */

double nn::mse_loss(double* (&Y), int dim)
{
    double l = 0.0;                                                         /// Initializes loss variable (accumulator)
#pragma omp parallel for num_threads(N_THREADS) reduction(+ : l) schedule(runtime)
    for (int i = 0; i < dim; i += 1)
    {
        l += (1.0 / 2.0) * (Y[i] - a[layers.size() - 1][i]) * (Y[i] - a[layers.size() - 1][i]);
    }

    return l;
}
