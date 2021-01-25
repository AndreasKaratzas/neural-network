
#include "Loss.h"

/**
 * Computes the model's MSE loss.
 * 
 * @param[in] y_pred the results computed by the model given an input
 * @param[in] y_real the expected output. This is the ground truth given the same input
 * @param[in] out_dim the dimensions of the given vectors
 * 
 * @note The given vectors must be of the same dimensions.
 */
long double mse_loss(long double* y_pred, long double* y_real, int out_dim)
{
    int i;
    long double loss = 0.0;
    
#pragma omp parallel for num_threads(N_THREADS) reduction(+ : loss) schedule(runtime) default(none) private(i) shared(loss, y_pred, y_real, out_dim)
    for (i = 0; i < out_dim; i += 1)
    {
        loss += (1.0 / 2.0) * (y_pred[i] - y_real[i]) * (y_pred[i] - y_real[i]);
    }

    return loss;
}
