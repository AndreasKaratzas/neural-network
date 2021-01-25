
#include "Neural.h"

/**
 * Generates a random long double precision floating point random number.
 * The number belongs in a subset defined by the user.
 * 
 * @param[in] min the lower bound of the subset of the randomly generated number
 * @param[in] max the upper bound of the subset of the randomly generated number
 * 
 * @return a random and bounded long double floating point number
 */
long double rand_probability(long double min, long double max)
{
    return (long double)(((rand() + 0.0) / (RAND_MAX + 0.0)) * (max - min)) + (min);
}

/**
 * Setter for the `num_layers` attribute of the neural network instance.
 * 
 * @param[in, out] nn the neural network instance
 * @param[in] L the number of layers to set the `num_layers` attribute
 */
void set_num_layers(MLP* nn, int L)
{
    nn->num_layers = L;
}

/**
 * Setter for the `filter` attribute of the neural network instance.
 * This attribute is known as the neurons' activation function.
 * 
 * @param[in, out] nn the neural network instance
 * @param[in] activation function pointer to the activation function
 */
void set_filter(MLP* nn, long double (*(activation))(long double x))
{
    nn->filter = activation;
}

/**
 * Setter for the `autograd` attribute of the neural network instance.
 * This attribute is the gradient of the neurons' activation function.
 *
 * @param[in, out] nn the neural network instance
 * @param[in] derivative function pointer to the derivative of the activation function
 */
void set_autograd(MLP* nn, long double (*(derivative))(long double x))
{
    nn->autograd = derivative;
}


/**
 * Setter for the `sizes` attribute of the neural network instance.
 * This is the container of the layers' sizes.
 *
 * @param[in, out] nn the neural network instance
 * @param[in] layer_count the number of layers for that instance
 * @param[in] args the vector with the sizes of the model's layers
 */
void set_sizes(MLP* nn, int layer_count, int* args)
{
    int i;

    nn->sizes = (int*)malloc(layer_count * sizeof(int));

    assert(nn->sizes);

    for (i = 0; i < layer_count; i += 1)
    {
        nn->sizes[i] = args[i];
    }
}

/**
 * Setter for the `weights` attribute of the neural network instance.
 * This initializes the weights of all neurons' synapses.
 *
 * @param[in, out] nn the neural network instance
 * @param[in] layer_count the number of layers for that instance
 * @param[in] args the vector with the sizes of the model's layers
 */
void set_weights(MLP* nn, int layer_count, int* args)
{
    int i, j, k;

    while (NULL == (nn->weights = (long double***)malloc((layer_count - 1) * sizeof(long double**)))) {}

    for (i = 0; i < layer_count - 2; i += 1)                                                                            /// Weight initializer for all hidden layers
    {
        while (NULL == (nn->weights[i] = (long double**)malloc((args[i + 1] - 1) * sizeof(long double*)))) {}

        for (j = 0; j < args[i + 1] - 1; j += 1)                                                                        /// Do not connect the bias at layer `l` with any neuron at layer `l - 1`
        {
            while (NULL == (nn->weights[i][j] = (long double*)malloc(args[i] * sizeof(long double)))) {}

            for (k = 0; k < args[i]; k += 1)
            {
                nn->weights[i][j][k] = rand_probability(-1.0, 1.0);
            }
        }
    }

    while (NULL == (nn->weights[layer_count - 2] = (long double**)malloc(args[layer_count - 1] * sizeof(long double*)))) {}

    for (j = 0; j < args[layer_count - 1]; j += 1)                                                                      /// Weight initializer the last layer's synapses
    {
        while (NULL == (nn->weights[layer_count - 2][j] = (long double*)malloc(args[layer_count - 2] * sizeof(long double)))) {}

        for (k = 0; k < args[layer_count - 2]; k += 1)                                                                  /// There is no bias in the output layer
        {
            nn->weights[layer_count - 2][j][k] = rand_probability(-1.0, 1.0);
        }
    }
}


/**
 * Destructor for the weights of the synapses given a model.
 *
 * @param[in, out] nn the neural network instance
 */
void delete_weights(MLP* nn)
{
    int i, j;

    for (i = 0; i < nn->num_layers - 1; i += 1)
    {
        for (j = 0; j < nn->sizes[i + 1]; j += 1)
        {
            free(nn->weights[i][j]);
        }
        free(nn->weights[i]);
    }
    free(nn->weights);
}

/**
 * Initializes a neural network instance.
 * 
 * @param[in, out] nn the neural network instance
 * @param[in] args the vector with the sizes of the model's layers 
 * @param[in] layer_count the number of layers for that instance
 * @param[in] activation function pointer to the activation function
 * @param[in] derivative function pointer to the derivative of the activation function
 */
void compile_nn(MLP* nn, int* args, int layer_count, long double (*(activation))(long double x), long double (*(derivative))(long double x))
{
    set_num_layers(nn, layer_count);
    set_sizes(nn, layer_count, args);
    set_filter(nn, activation);
    set_autograd(nn, derivative);
    set_weights(nn, layer_count, args);
}

/**
 * Deletes all memory allocated for a neural network.
 * 
 * param[in, out] nn the neural network instance
 */
void delete_nn(MLP* nn)
{
    delete_weights(nn);
}
