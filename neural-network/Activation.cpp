
#include "Activation.hpp"

/**
 * Filters the input using the sigmoid activation function.
 *
 * @param[in] x this is the double precision floating
 *              point variable to be filtered by the sigmoid
 *
 * @return the filtered value
 */
double sigmoid(double x)
{
    return 1.0 / (1.0 + std::pow(EXP, -x)); /// Sigmoid formula
}

/**
 * Computes the derivative of the sigmoid function.
 * This implementation uses already filtered values
 * by the sigmoid to speed up the computation process.
 *
 * @param[in] x this is the double precision floating
 *              point variable to be differentiated
 *
 * @return the derivative of x with respect to the sigmoid function
 */
double sig_derivative(double x)
{
    return (x * (1.0 - x));                 /// Sigmoid derivative formula
}

/**
 * Filters the input using the ReLU activation function.
 *
 * @param[in] x this is the double precision floating
 *              point variable to be filtered by the ReLU
 *
 * @return the filtered value
 */
double relu(double x)
{
    return (x > 0.0 ? x : 0.0);             /// ReLU formula
}

/**
 * Computes the derivative of the ReLU function.
 *
 * @param[in] x this is the double precision floating
 *              point variable to be differentiated
 *
 * @return the derivative of x with respect to the ReLU function
 */
double rel_derivative(double x)
{
    return (x < 0.0 ? 0.0 : 1.0);           /// ReLU derivative formula
}
