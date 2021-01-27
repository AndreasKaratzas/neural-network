
#include "Neural.hpp"

int nn::get_label(double* (&y_pred))
{
    int label;
    double max_val = -2.0;

    for (int i = 0; i < layers.back(); i += 1)
    {
        if (y_pred[i] > max_val)
        {
            max_val = y_pred[i];
            label = i;
        }
    }

    return label;
}

int nn::predict(double* (&X), double* (&Y))
{
    zero_grad(X);
    forward();
    return get_label(z[layers.size() - 1]);
}


void nn::set_layers(const std::vector<int>& l)
{
    for (auto& elem : l)
    {
        layers.push_back(elem);
    }
}

/**
 * Allocates memory space for the dynamic matrix that contains the neurons' unfiltered value.
 *
 * @param[in, out] l the neural network layer structure vector
 *
 * @note    The `z` container for each neuron `i` in layer a `u` holds the sum given by the
 *          formula:
 *          \sum_j^L{synapse_{i, j} * value_{j}}, where synapse is the numerical weight
 *          of the synapse between neuron `i`, `j` and the value of `j` is the filtered
 *          output of that neuron and `L` is the number of neurons found in layer `u - 1`.
 *          The filter refers to the activation function used to 'activate' the neurons
 *          in the neural network.
 */
void nn::set_z(const std::vector<int>& l)
{
    z = new double* [l.size()];
    for (int i = 0; i < l.size(); i += 1)
    {
        z[i] = new double[l[i]];
    }
}

/**
 * Allocates memory space for the dynamic matrix that contains the neurons' filtered value.
 *
 * @param[in, out] l the neural network layer structure vector
 *
 * @note    The `a` container for each neuron `i` in layer `l` holds the sum given by the
 *          formula:
 *          f{(z_i)}, \forall i \in `l`, where f is the chosen activation function for every
 *          neuron i nthe model.
 */
void nn::set_a(const std::vector<int>& l)
{
    a = new double* [l.size()];
    for (int i = 0; i < l.size(); i += 1)
    {
        a[i] = new double[l[i]];
    }
}

/**
 * Allocates memory space for the dynamic matrix that contains the neurons' error.
 *
 * @param[in, out] l the neural network layer structure vector
 */
void nn::set_delta(const std::vector<int>& l)
{
    delta = new double* [l.size() - 1];
    for (int i = 1; i < l.size(); i += 1)
    {
        delta[i - 1] = new double[l[i]];
    }
}

void nn::set_weights(const std::vector<int>& l, const double min, const double max)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(min, max);

    weights = new double** [l.size() - 1];
    for (int i = 1; i < l.size() - 1; i += 1)
    {
        weights[i - 1] = new double* [l[i] - 1];
        for (int j = 0; j < l[i] - 1; j += 1)
        {
            weights[i - 1][j] = new double[l[i - 1]];
            for (int k = 0; k < l[i - 1]; k += 1)
            {
                weights[i - 1][j][k] = dist(gen);
            }
        }
    }
    weights[l.size() - 2] = new double* [l.back()];
    for (int j = 0; j < l.back(); j += 1)
    {
        weights[l.size() - 2][j] = new double[l[l.size() - 2]];
        for (int k = 0; k < l[l.size() - 2]; k += 1)
        {
            weights[l.size() - 2][j][k] = dist(gen);
        }
    }
}

void nn::compile(const std::vector<int>& l, const double min, const double max)
{
    set_layers(l);
    set_z(l);
    set_a(l);
    set_delta(l);
    set_weights(l, min, max);
}

/**
 * Clears the past values computed created during feed forward and/or back propagation processes.
 *
 * @param[in, out] X a vector that has been initialized with a random sample from the training data subset
 *
 * @note This function is called upon both training and evaluation. That's why it also clears past values of the `delta` container.
 */
void nn::zero_grad(double* (&X))
{
    for (int i = 0; i < layers.size(); i += 1)
    {
        if (i == 0)
        {
            for (int j = 0; j < layers[i] - 1; j += 1)
            {
                z[i][j] = X[j];
                a[i][j] = X[j];
            }
            z[i][layers[i] - 1] = 1.0;
            a[i][layers[i] - 1] = 1.0;
        }
        else if (i == layers.size() - 1)
        {
            for (int j = 0; j < layers[i]; j += 1)
            {
                z[i][j] = 0.0;
                a[i][j] = 0.0;
                delta[i - 1][j] = 0.0;
            }
        }
        else
        {
            for (int j = 0; j < layers[i] - 1; j += 1)
            {
                z[i][j] = 0.0;
                a[i][j] = 0.0;
                delta[i - 1][j] = 0.0;
            }
            z[i][layers[i] - 1] = 1.0;
            a[i][layers[i] - 1] = 1.0;
            delta[i - 1][layers[i] - 1] = 0.0;
        }
    }
}
