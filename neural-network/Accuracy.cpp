
#include "Accuracy.h"

/**
 * Computes model accuracy. This function pools the element with the maximum value from the 
 * predictions vector and then uses the target vector to compute the accuracy of the given
 * prediction.
 * 
 * @param[in] y_pred pointer to the vector with the values predicted by the model
 * @param[in] y_target pointer to the vector with the desired values
 * @param[in] dim number of the vectors' elements
 * 
 * @return 1 if the element with the maximum value from the predictions vector was accurate, else 0
 * 
 * @note the given vectors must be of the same length
 */
int model_accuracy(long double* y_pred, long double* y_target, int dim)
{
    int neuron, target_idx, pred_idx = -1, accuracy = 0, max_val = -2.0;

    assert(array_sizeof(y_pred) == array_sizeof(y_target));                 /// Mask vector compatibility failure

    for (neuron = 0; neuron < dim; neuron += 1)                             /// Iterate through the vector with the predictions
    {
        if (max_val < y_pred[neuron])                                       /// Find the neuron with the maximum (filtered) value
        {
            max_val = y_pred[neuron];
            pred_idx = neuron;
        }
    }
    
    return (pred_idx < 0 ? 0 : y_target[pred_idx] > 0.9 ? 1 : 0);           /// Computes the accuracy of the given prediction based on the elite neuron found after the previous iteration
}
