
#include "Activation.h"

/**
 * Filters the input using the sigmoid activation function. 
 * 
 * @param[in] x this is the long double precision floating
 *              point variable to be filtered by the sigmoid
 * @return the filtered value 
 */
long double sigmoid(long double x)
{
    if (x > 13.0)                           /// Defines upper limit for the sigmoid to prevent overflow
    {
        return 1.0;
    }
    else if (x < -13.0)                     /// Defines lower limit for the sigmoid to prevent underflow
    {
        return 0.0;
    }
    else
    {
        return (1.0 / (1.0 + exp(-x)));     /// Sigmoid formula
    }
}

/**
 * Computes the derivative of the sigmoid function.
 * This implementation uses already filtered values 
 * by the sigmoid to speed up the computation process.
 *
 * @param[in] x this is the long double precision floating
 *              point variable to be differentiated
 * @return the derivative of x with respect to the sigmoid function
 */
long double sig_derivative(long double x)
{
    return (x * (1.0 - x));                 /// Sigmoid derivative formula
}

/**
 * Filters the input using the ReLU activation function.
 *
 * @param[in] x this is the long double precision floating
 *              point variable to be filtered by the ReLU
 * @return the filtered value
 */
long double relu(long double x)
{
    return (x > 0.0 ? x : 0.0);             /// ReLU formula
}

/**
 * Computes the derivative of the ReLU function.
 *
 * @param[in] x this is the long double precision floating
 *              point variable to be differentiated
 * @return the derivative of x with respect to the ReLU function
 */
long double rel_derivative(long double x)
{
    return (x < 0.0 ? 0.0 : 1.0);           /// ReLU derivative formula
}
