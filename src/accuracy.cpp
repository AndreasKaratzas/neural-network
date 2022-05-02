
#include "neural.hpp"

/**
 * Computes model accuracy. This function pools the element with the maximum value from the
 * predictions vector and then uses the target vector to compute the accuracy of the given
 * prediction.
 *
 * @param[in, out] Y the vector with the desired values
 * @param[in] dim number of the vectors' elements
 *
 * @return 1 if the element with the maximum value from the predictions vector was accurate, else 0
 *
 * @note the given vectors must be of the same length
 * 
 * @note Although passed by reference, the `Y` placeholder is not altered.
 */
int nn::accuracy(double* (&Y), int dim)
{
    double max_val = -2.0;
    int max_idx = 0;

    for (int i = 0; i < dim; i += 1)                        /// Iterate through the vector with the predictions
    {
        if (a[layers.size() - 1][i] > max_val)              /// Find the neuron with the maximum (filtered) value
        {
            max_val = a[layers.size() - 1][i];
            max_idx = i;
        }
    }

    return Y[max_idx] > 0.9 ? 1 : 0;                        /// Computes the accuracy of the given prediction based on the elite neuron found after the previous iteration
}