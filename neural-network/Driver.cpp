
#include "Driver.h"

/**
 * Implements the driver for the Neural Network.
 *
 * @param[in] argc number of user arguments
 * @param[in] argv vector of user arguments
 * 
 * @return 0, if the executable was terminated normally
 * 
 * @note    For the driver to work properly, adjust the project settings found at the `Common.h` file.
 *          One such adjustment is to define the filepaths of the training and the evaluation subsets.
 *          Another stronlgy recommended change is the number of threads requested by the OS. This number
 *          is recommended to be equal to the number of the hosts's Logical Processors. This will very 
 *          possibly optimize execution time and therefore increase performance.
 */
int main(int argc, char* argv[])
{
    srand((unsigned int)time(NULL));                                                                /// Seeds the random generator
    
    MLP fcn;                                                                                        /// Declares the image of the neural network
    int *hidden_size, *args, *train_dim, *test_dim, *train_accuracy, input_size, output_size, hidden_dim, filter;
    char *activation;
    double start, end;
    long double **x_train, **y_train, **x_test, **y_test, *train_loss, test_loss;
    long double (*(activations[N_ACTIVATIONS]))(long double x) = { sigmoid, relu };                 /// Pointers to all the available activation functions of the project
    long double (*(derivatives[N_ACTIVATIONS]))(long double x) = { sig_derivative, rel_derivative };/// Pointers to the corresponding derivatives of the project's activation functions

    hidden_size = (int*)malloc(1 * sizeof(int));                                                    /// Initializes the image of the model's hidden layer 
    activation = (char*)malloc(1 * sizeof(char));                                                   /// Initializes the image of the model's activation function
    train_dim = (int*)malloc(2 * sizeof(int));                                                      /// Initialises the image of the training dataset's dimensions
    test_dim = (int*)malloc(2 * sizeof(int));                                                       /// Initialises the image of the evaluation dataset's dimensions

    assert(hidden_size);                                                                            /// Masks memory allocation fault
    assert(activation);                                                                             /// Masks memory allocation fault
    assert(train_dim);                                                                              /// Masks memory allocation fault
    assert(test_dim);                                                                               /// Masks memory allocation fault

    memset(train_dim, -1, 2 * sizeof(int));                                                         /// Sets the training dataset's dimensions
    memset(test_dim, -1, 2 * sizeof(int));                                                          /// Sets the evaluation dataset's dimensions

    start = omp_get_wtime();                                                                        /// Initializes benchmark
    hidden_dim = parse_arguments(argc, argv, &input_size, hidden_size, &output_size, activation);   /// Parses user arguments
    filter = get_nn_activation(activation);                                                         /// Initialises the filter variable with the networks activation function
    get_args(args, input_size, hidden_size, output_size, hidden_dim);                               /// Wraps the user's argumetns around a vector, called `args`
    print_parser_results(input_size, hidden_size, hidden_dim, output_size, activation);             /// Prints the user's arguments
    compile_nn(&fcn, args, hidden_dim + 2, activations[filter], derivatives[filter]);               /// Initializes the neural network's image
    train_test_split(x_train, y_train, train_dim, x_test, y_test, test_dim);                        /// Loads the MNIST dataset

    assert(train_dim[0] > 0 && train_dim[1] > 0 && test_dim[0] > 0 && test_dim[1] > 0);             /// Masks dataset loading failure
    assert(train_dim[1] == fcn.sizes[0] - 1);                                                       /// Masks model - dataset compatibility fault
    assert(fcn.sizes[fcn.num_layers - 1] == MNIST_CLASSES);                                         /// Masks model - dataset compatibility fault

    train(&fcn, x_train, y_train, train_loss, train_accuracy, train_dim);                           /// Trains the model
    test_loss = test(&fcn, x_test, y_test, test_dim);                                               /// Evaluates the model
    deallocate_dataset(x_train, y_train, x_test, y_test, train_dim, test_dim);                      /// Deallocates memory
    end = omp_get_wtime();                                                                          /// Terminates the benchmark
    printf("\nBenchmark results: %lf\n", end - start);                                              /// Prints benchmark results

    return(0);
}