
#include "Activation.hpp"

/** 
 * Efficient implementation to calculate e raise to the power x. 
 * 
 * @param[in] x the numerical base of the power
 * @param[in] y the numerical exponent of the power
 * 
 * @return the double precision floating point result of the power operation.
 * 
 * @note Returns approximate value of e^x using sum of first n terms of Taylor Series.
 * 
 * @remark https://www.geeksforgeeks.org/program-to-efficiently-calculate-ex/
 */
double exp(double x)
{
    double sum = 1.0;
    int precision = 100;

    for (int i = precision - 1; i > 0; i -= 1)
    {
        sum = 1.0 + x * sum / (double)i;
    }

    return sum;
}

/**
 * Filters the input using the sigmoid activation function.
 * This implementation exploits Taylor series for fast computation.
 *
 * @param[in] x this is the double precision floating
 *              point variable to be filtered by the sigmoid
 *
 * @return the filtered value
 * 
 * @note    The implementation does not utilize the std::exp routine to
 *          avoid overflow failures.
 */
double fast_sigmoid(double x)
{
    if (x > 13.0)
    {
        return 1.0;
    }
    else if (x < -13.0)
    {
        return 0.0;
    }
    else
    {
        return 1.0 / (1.0 + exp(-x));       /// Sigmoid formula
    }
}

/**
 * Safe implementation of sigmoid. This implementation gives a more precise result.
 * 
 * @param[in] x this is the double precision floating
 *              point variable to be filtered by the sigmoid
 * 
 * @return the filtered value
 * 
 * @note    The std::pow() function is low on performance.
 */
double sigmoid(double x)
{
    if (x > 45.0)
    {
        return 1.0;
    }
    else if (x < -45.0)
    {
        return 0.0;
    }
    else
    {
        return 1.0 / (1.0 + std::pow(EXP, -x));
    }
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
