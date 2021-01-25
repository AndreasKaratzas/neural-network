
#include "Utility.h"

/**
 * Decodes the user input regarding the activation function of the model.
 * 
 * @param[in] activation the string provided by the user describing the model's activation function
 * 
 * @return the ID correspinding to the chosen activation function
 */
int get_nn_activation(char* activation)
{
    return strcmp(activation, "sigmoid") == 0 ? 0 : strcmp(activation, "relu") == 0 ? 1 : -1;
}

/**
 * Wraps the user arguments regarding the model's structure (neural network's layers) around a single container.
 * 
 * @param[in, out] args the container to wrap the given data
 * @param[in] input_layer the number of neurons in the first (input) layer of the neural network
 * @param[in] hidden_layer the vector with the number of neurons in the hidden layers of the neural network
 * @param[in] output_layer the number of neurons in the last (output) layer of the neural network
 * @param[in] hid_dim the number hidden layers in the neural network
 */
void get_args(int* (&args), int input_layer, int *hidden_layer, int output_layer, int hid_dim)
{
    int i;

    args = (int*)malloc((hid_dim + 2) * sizeof(int));                                               /// Allocates enough memory space for the container

    assert(args);                                                                                   /// Masks memory allocation fault

    args[0] = input_layer + 1;

    for (i = 0; i < hid_dim; i += 1)
    {
        args[i + 1] = hidden_layer[i] + 1;
    }

    args[hid_dim + 1] = output_layer;
}

/**
 * Allocates memory space for the dynamic matrix that contains the neurons' error.
 * 
 * @param[in, out] delta the dynamic matrix that contains the neurons' error
 * @param[in, out] fcn the neural network image
 * 
 * @note    The neural network must be an argument because the dimesions of the `delta` container
 *          will be determined based on the model's layer sizes.
 */
void allocate_delta(long double** (&delta), MLP* fcn)
{
    int layer, neuron;

    while (NULL == (delta = (long double**)malloc((fcn->num_layers - 1) * sizeof(long double*)))) {}

    for (layer = 0; layer < fcn->num_layers - 2; layer += 1)
    {
        while (NULL == (delta[layer] = (long double*)malloc(fcn->sizes[layer + 1] * sizeof(long double)))) {}

        for (neuron = 0; neuron < fcn->sizes[layer + 1]; neuron += 1)
        {
            delta[layer][neuron] = 0.0;                                                             /// Error is initialized with 0 (zero)
        }
    }

    while (NULL == (delta[fcn->num_layers - 2] = (long double*)malloc(fcn->sizes[fcn->num_layers - 1] * sizeof(long double)))) {}

    for (neuron = 0; neuron < fcn->sizes[fcn->num_layers - 1]; neuron += 1)
    {
        delta[fcn->num_layers - 2][neuron] = 0.0;                                                   /// Error is initialized with 0 (zero)
    }
}

/**
 * Allocates memory space for the dynamic matrix that contains the neurons' unfiltered value.
 *
 * @param[in, out] Z the dynamic matrix that contains the neurons' unfiltered value
 * @param[in, out] fcn the neural network image
 * @param[in] x_train a vector that has been initialized with a random sample from the training data subset
 *
 * @note    The `Z` container for each neuron `i` in layer `l` holds the sum given by the
 *          formula:
 *          \sum_j^L{synapse_{i, j} * value_{j}}, where synapse is the numerical weight 
 *          of the synapse between neuron `i`, `j` and the value of `j` is the filtered
 *          output of that neuron and `L` is the number of neurons found in layer `l - 1`.
 *          The filter refers to the activation function used to 'activate' the neurons 
 *          in the neural network.
 * 
 * @note    The neural network given as argument is not altered.
 */
void allocate_Z(long double** (&Z), MLP* fcn, long double* x_train)
{
    int layer, neuron;

    while (NULL == (Z = (long double**)malloc(fcn->num_layers * sizeof(long double*)))) {}

    while (NULL == (Z[0] = (long double*)malloc(fcn->sizes[0] * sizeof(long double)))) {}

    for (neuron = 0; neuron < fcn->sizes[0] - 1; neuron += 1)
    {
        Z[0][neuron] = x_train[neuron];
    }

    Z[0][fcn->sizes[0] - 1] = 1.0;

    for (layer = 1; layer < fcn->num_layers - 1; layer += 1)
    {
        while (NULL == (Z[layer] = (long double*)malloc(fcn->sizes[layer] * sizeof(long double)))) {}

        for (neuron = 0; neuron < fcn->sizes[layer] - 1; neuron += 1)
        {
            Z[layer][neuron] = 0.0;
        }
        Z[layer][fcn->sizes[layer] - 1] = 1.0;
    }

    while (NULL == (Z[fcn->num_layers - 1] = (long double*)malloc(fcn->sizes[fcn->num_layers - 1] * sizeof(long double)))) {}

    for (neuron = 0; neuron < fcn->sizes[fcn->num_layers - 1]; neuron += 1)
    {
        Z[fcn->num_layers - 1][neuron] = 0.0;
    }
}

/**
 * Allocates memory space for the dynamic matrix that contains the neurons' filtered value.
 *
 * @param[in, out] A the dynamic matrix that contains the neurons' filtered value
 * @param[in, out] fcn the neural network image
 * @param[in] x_train a vector that has been initialized with a random sample from the training data subset
 *
 * @note    The `A` container for each neuron `i` in layer `l` holds the sum given by the
 *          formula:
 *          f{(Z_i)}, \forall i \in `l`, where f is the chosen activation function for every
 *          neuron i nthe model.
 * 
 * @note    The neural network given as argument is not altered.
 */
void allocate_A(long double** (&A), MLP* fcn, long double* x_train)
{
    int layer, neuron;

    while (NULL == (A = (long double**)malloc(fcn->num_layers * sizeof(long double*)))) {}

    while (NULL == (A[0] = (long double*)malloc(fcn->sizes[0] * sizeof(long double)))) {}

    for (neuron = 0; neuron < fcn->sizes[0] - 1; neuron += 1)
    {
        A[0][neuron] = x_train[neuron];
    }

    A[0][fcn->sizes[0] - 1] = 1.0;

    for (layer = 1; layer < fcn->num_layers - 1; layer += 1)
    {
        while (NULL == (A[layer] = (long double*)malloc(fcn->sizes[layer] * sizeof(long double)))) {}

        for (neuron = 0; neuron < fcn->sizes[layer] - 1; neuron += 1)
        {
            A[layer][neuron] = 0.0;
        }
        A[layer][fcn->sizes[layer] - 1] = 1.0;
    }

    while (NULL == (A[fcn->num_layers - 1] = (long double*)malloc(fcn->sizes[fcn->num_layers - 1] * sizeof(long double)))) {}

    for (neuron = 0; neuron < fcn->sizes[fcn->num_layers - 1]; neuron += 1)
    {
        A[fcn->num_layers - 1][neuron] = 0.0;
    }
}

/**
 * Clears the past values computed during feed forward and back propagation processes.
 * 
 * @param[in, out] delta the dynamic matrix that contains the neurons' error
 * @param[in, out] fcn the neural network image
 * @param[in, out] Z the dynamic matrix that contains the neurons' unfiltered value
 * @param[in, out] A the dynamic matrix that contains the neurons' filtered value
 * @param[in] x_train a vector that has been initialized with a random sample from the training data subset
 * 
 * @note The neural network given as argument is not altered.
 * 
 * @note This function is called upon training. That's why it also clears past values of the `delta` container.
 */
void zero_grad(long double** (&delta), MLP* fcn, long double** (&Z), long double** (&A), long double* x_train)
{
    int layer, neuron;

    for (neuron = 0; neuron < fcn->sizes[0] - 1; neuron += 1)                                       /// Loops through all neurons of the first layer
    {
        Z[0][neuron] = x_train[neuron];                                                             /// Initializes neurons' values with the given training sample
        A[0][neuron] = x_train[neuron];
    }

    Z[0][fcn->sizes[0] - 1] = 1.0;                                                                  /// Initializes bias value 
    A[0][fcn->sizes[0] - 1] = 1.0;

    for (layer = 1; layer < fcn->num_layers - 1; layer += 1)                                        /// Loops through all hidden layers
    {
        for (neuron = 0; neuron < fcn->sizes[layer] - 1; neuron += 1)
        {
            delta[layer - 1][neuron] = 0.0;                                                         /// Deletes (initializes with zero) previously computed data
            Z[layer][neuron] = 0.0;
            A[layer][neuron] = 0.0;
        }
        delta[layer - 1][fcn->sizes[layer] - 1] = 0.0;                                              /// Clears bias
        Z[layer][fcn->sizes[layer] - 1] = 1.0;                                                      /// Resets bias
        A[layer][fcn->sizes[layer] - 1] = 1.0;
    }

    for (neuron = 0; neuron < fcn->sizes[fcn->num_layers - 1]; neuron += 1)                         /// Loops through all neurons found in the output layer of the model
    {
        delta[fcn->num_layers - 2][neuron] = 0.0;                                                   /// There is no bias here
        Z[fcn->num_layers - 1][neuron] = 0.0;
        A[fcn->num_layers - 1][neuron] = 0.0;
    }
}

/**
 * Clears the past values computed only during feed forward process.
 *
 * @param[in, out] fcn the neural network image
 * @param[in, out] Z the dynamic matrix that contains the neurons' unfiltered value
 * @param[in, out] A the dynamic matrix that contains the neurons' filtered value
 * @param[in] x_train a vector that has been initialized with a random sample
 *
 * @note    The neural network given as argument is not altered.
 *
 * @note    This function is called after the model has been trained. Before prediction, this function
 *          function is called to clear previously computed data.
 * 
 * @note    This function oveloads `zero_grad` function defined above
 */
void zero_grad(MLP* fcn, long double** (&Z), long double** (&A), long double* x_train)
{
    int layer, neuron;

    for (neuron = 0; neuron < fcn->sizes[0] - 1; neuron += 1)
    {
        Z[0][neuron] = x_train[neuron];
        A[0][neuron] = x_train[neuron];
    }

    Z[0][fcn->sizes[0] - 1] = 1.0;
    A[0][fcn->sizes[0] - 1] = 1.0;

    for (layer = 1; layer < fcn->num_layers - 1; layer += 1)
    {
        for (neuron = 0; neuron < fcn->sizes[layer] - 1; neuron += 1)
        {
            Z[layer][neuron] = 0.0;
            A[layer][neuron] = 0.0;
        }
        Z[layer][fcn->sizes[layer] - 1] = 1.0;
        A[layer][fcn->sizes[layer] - 1] = 1.0;
    }

    for (neuron = 0; neuron < fcn->sizes[fcn->num_layers - 1]; neuron += 1)
    {
        Z[fcn->num_layers - 1][neuron] = 0.0;
        A[fcn->num_layers - 1][neuron] = 0.0;
    }
}

/**
 * Releases the memory space allocated for the `Z` container.
 * That container was used to store the neurons' unfiltered values.
 * 
 * @param[in, out] Z the dynamic matrix that contains the neurons' unfiltered value
 * @param[in, out] fcn the neural network image
 * 
 * @note The neural network given as argument is not altered.
 */
void deallocate_Z(long double** (&Z), MLP* fcn)
{
    int layer;

    for (layer = 0; layer < fcn->num_layers; layer += 1)
    {
        free(Z[layer]);
    }

    free(Z);
}

/**
 * Releases the memory space allocated for the `A` container.
 * That container was used to store the neurons' filtered values.
 *
 * @param[in, out] A the dynamic matrix that contains the neurons' filtered value
 * @param[in, out] fcn the neural network image
 *
 * @note The neural network given as argument is not altered.
 */
void deallocate_A(long double** (&A), MLP* fcn)
{
    int layer;

    for (layer = 0; layer < fcn->num_layers; layer += 1)
    {
        free(A[layer]);
    }

    free(A);
}

/**
 * Releases the memory space allocated for the `delta` container.
 * That container was used to store the neurons' errors during
 * model's training.
 *
 * @param[in, out] delta the dynamic matrix that contains the neurons' error
 * @param[in, out] fcn the neural network image
 *
 * @note The neural network given as argument is not altered.
 */
void deallocate_delta(long double** (&delta), MLP* fcn)
{
    int layer;

    for (layer = 0; layer < fcn->num_layers - 1; layer += 1)
    {
        free(delta[layer]);
    }

    free(delta);
}

/**
 * Releases the memory space allocated for the dataset.
 *
 * @param[in, out] x_train the matrix that holds all training samples
 * @param[in, out] y_train the matrix that holds the corresponding (expected) outputs
 * @param[in, out] x_test the matrix that holds all evaluation samples
 * @param[in, out] y_test the matrix that holds the corresponding (expected) outputs
 * @param[in] train_dim the shape of the `x_train` matrix
 * @param[in] test_dim the shape of the `x_test` matrix
 *
 * @note The neural network given as argument is not altered.
 */
void deallocate_dataset(long double** (&x_train), long double** (&y_train), long double** (&x_test), long double** (&y_test), int* train_dim, int* test_dim)
{
    int row;

    for (row = 0; row < train_dim[0]; row += 1)
    {
        free(x_train[row]);
        free(y_train[row]);
    }

    free(x_train);
    free(y_train);

    for (row = 0; row < test_dim[0]; row += 1)
    {
        free(x_test[row]);
        free(y_test[row]);
    }

    free(x_test);
    free(y_test);
}
