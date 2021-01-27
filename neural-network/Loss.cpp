
#include "Neural.hpp"

/**
 * Computes the model's MSE loss.
 *
 * @param[in] Y the expected output. This is the ground truth given the same input
 *
 * @note The given vectors must be of the same dimensions.
 */

double nn::mse_loss(double* (&Y), int c)
{
    double l = 0.0;
#pragma omp parallel for num_threads(N_THREADS) reduction(+ : l) schedule(runtime)
    for (int i = 0; i < c; i += 1)
    {
        l += (1.0 / 2.0) * (Y[i] - a[layers.size() - 1][i]) * (Y[i] - a[layers.size() - 1][i]);
    }

    return l;
}