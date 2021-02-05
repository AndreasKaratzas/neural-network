
#include "Neural.hpp"

/**
 * This function pools the element with the maximum value from the
 * predictions vector.
 *
 * @param[in, out] y_pred the predictions vector
 *
 * @return the index of the element with the maximum value
 * 
 * @note Although passed by reference, `y_pred` is not altered.
 */
int nn::get_label(double* (&y_pred))
{
    int label;
    double max_val = -2.0;

    for (int i = 0; i < layers[layers.size() - 1]; i += 1)
    {
        if (y_pred[i] > max_val)
        {
            max_val = y_pred[i];
            label = i;
        }
    }

    return label;
}

/**
 * Uses model to make predictions using custom inputs.
 * 
 * @param[in, out] X the input to be given to the model
 * 
 * @return the predicted class with respect to the given input
 * 
 * @note Although passed by reference, the `X` placeholder is not altered.
 */
int nn::predict(double* (&X))
{
    zero_grad(X);
    forward();
    return get_label(a[layers.size() - 1]);
}

/**
 * Initializes model structure.
 * 
 * @param[in] l the vector containing the model's structure
 */
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
 *          neuron i the model.
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

/**
 * Sets model's weights of synapses.
 * 
 * @param[in] l the neural network layer structure vector
 * @param[in] min the minimum weight of a synapse
 * @param[in] max the maximum weight of a synapse
 */
void nn::set_weights(const std::vector<int>& l, const double min, const double max)
{
    std::random_device rd;                                              /// Initializes non-deterministic random generator
    std::mt19937 gen(rd());                                             /// Seeds mersenne twister
    std::uniform_real_distribution<> dist(min, max);                    /// Distribute results between `min` and `max` inclusive

    weights = new double** [l.size() - 1];                              /// Allocates memory for the weights container
    for (int i = 1; i < l.size() - 1; i += 1)
    {
        weights[i - 1] = new double* [l[i] - 1];                        /// Allocates memory for the weights of a layer in a neural network
        for (int j = 0; j < l[i] - 1; j += 1)
        {
            weights[i - 1][j] = new double[l[i - 1]];                   /// Allocates memory for the weights of each neuron in a layer
            for (int k = 0; k < l[i - 1]; k += 1)
            {
                weights[i - 1][j][k] = dist(gen);                       /// Uses random generator to initialize synapse
            }
        }
    }
    weights[l.size() - 2] = new double* [l[l.size() - 1]];              /// Initializes weights in the output layer
    for (int j = 0; j < l[l.size() - 1]; j += 1)                        /// There is no bias in the output layer
    {
        weights[l.size() - 2][j] = new double[l[l.size() - 2]];         /// Allocates memory for the weights of each neuron in the output layer
        for (int k = 0; k < l[l.size() - 2]; k += 1)
        {
            weights[l.size() - 2][j][k] = dist(gen);                    /// Uses random generator to initialize synapse
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
    for (int j = 0; j < layers[0] - 1; j += 1)                          /// Prepare - initialize input layer
    {
        z[0][j] = X[j];
        a[0][j] = X[j];
    }
    z[0][layers[0] - 1] = 1.0;
    a[0][layers[0] - 1] = 1.0;

    for (int i = 1; i < layers.size() - 1; i += 1)                      /// Prepare - initialize hidden layers
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

    for (int j = 0; j < layers[layers.size() - 1]; j += 1)              /// Prepare - initialize output layer
    {
        z[layers.size() - 1][j] = 0.0;
        a[layers.size() - 1][j] = 0.0;
        delta[layers.size() - 2][j] = 0.0;
    }
}

/**
 * Prints Neural Network layer structure.
 */
void nn::summary(void)
{
    int l = 0;
    
    std::string s(CLI_WINDOW_WIDTH + 10, '-');

    std::cout << "\n\nNeural Network Summary:\t\t[f := Sigmoid]\n" << s << std::endl;
    
    for (auto& elem : layers)
    {
        std::cout << "Layer [" << ++l << "]\t" << std::setw(4) << elem << " neurons\n";
    }
}
