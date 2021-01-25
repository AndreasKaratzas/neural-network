
#include "Fit.h"

/**
 * Trains the given model. The model is a simple multi-
 * layer feed forward perceptron.
 * 
 * @param[in] fcn the `Fully Connected Network`
 * @param[in] x_train the input matrix used during training. Each row is an input vector for the neural network
 * @param[in] y_train the matrix with the corresponding expected output
 * @param[in, out] train_accuracy the vector with the model's accuracy throughout the different epochs
 * @param[in] train_dim the size of the training dataset. This is the number of rows of the `x_train` matrix
 */
void train(MLP* fcn, long double** (&x_train), long double** y_train, long double* (&train_loss), int* (&train_accuracy), int* (&train_dim))
{
    int epoch, sample_count, sample_idx;
    double start, end;
    long double **Z, **A, **delta;                                                          /// `Z[]` corresponds to `DL[]` and `A[]` to `OL[]`

    train_loss = (long double*)malloc(EPOCHS * sizeof(long double));                        /// Allocates memory for the container of the model's training loss
    train_accuracy = (int*)malloc(EPOCHS * sizeof(int));                                    /// Allocates memory for the container of the model's training accuracy 

    assert(train_loss);                                                                     /// Masks memory fault
    assert(train_accuracy);                                                                 /// Masks memory fault

    allocate_Z(Z, fcn, x_train[0]);                                                         /// Allocates memory and initializes the `Z` contatiner
    allocate_A(A, fcn, x_train[0]);                                                         /// Allocates memory and initializes the `A` contatiner
    allocate_delta(delta, fcn);

    for (epoch = 0; epoch < EPOCHS; epoch += 1)                                             /// Trains model
    {
        train_accuracy[epoch] = 0;                                                          /// Initializes epoch's training accuracy 
        train_loss[epoch] = 0.0;                                                            /// Initializes epoch's training loss

        start = omp_get_wtime();                                                            /// Benchmarks epoch
        for (sample_count = 0; sample_count < train_dim[0]; sample_count += 1)              /// Iterates through all examples of the training dataset
        {
            sample_idx = rand() % (int)MNIST_TRAIN;                                         /// Selects a random example
            zero_grad(delta, fcn, Z, A, x_train[sample_idx]);                               /// Resets the neurons of the neural network
            feedforward(fcn, Z, A);                                                         /// Feeds forward the selected input
            backprop(fcn, y_train[sample_idx], A, delta);                                   /// Computes the error for every neuron in the network
            optimize(fcn, delta, A);                                                        /// Optimizes weights using pack propagation
            train_accuracy[epoch] += model_accuracy(A[fcn->num_layers - 1], y_train[sample_idx], fcn->sizes[fcn->num_layers - 1]);
                                                                                            /// Updates epoch's accuracy of the model
            train_loss[epoch] += mse_loss(A[fcn->num_layers - 1], y_train[sample_idx], fcn->sizes[fcn->num_layers - 1]);
                                                                                            /// Updates epoch's accuracy of the model
        }
        end = omp_get_wtime();                                                              /// Terminates epoch's benchmark
        print_epoch_stats(epoch, (train_loss[epoch] / MNIST_TRAIN), train_accuracy[epoch], end - start);
                                                                                            /// Prints epoch's loss, accuracy and benchmark
    }

    deallocate_Z(Z, fcn);                                                                   /// Deallocates `Z` off memory
    deallocate_A(A, fcn);                                                                   /// Deallocates `A` off memory
    deallocate_delta(delta, fcn);                                                           /// Deallocates `delta` off memory
}

/**
 * Evaluates the given model. 
 * 
 * @param[in] fcn the `Fully Connected Network`
 * @param[in] x_test the input matrix used during evaluation. Each row is an input vector for the neural network
 * @param[in] y_test the matrix with the corresponding expected output
 * @param[in] test_dim the size of the evaluation dataset. This is the number of rows of the `x_test` matrix
 */
long double test(MLP* fcn, long double** (&x_test), long double** (&y_test), int* (&test_dim))
{
    int sample, test_accuracy = 0;
    double start, end;
    long double **Z, **A, test_loss = 0.0;

    allocate_Z(Z, fcn, x_test[0]);                                                          /// Allocates memory and initializes the `Z` contatiner
    allocate_A(A, fcn, x_test[0]);                                                          /// Allocates memory and initializes the `A` contatiner

    start = omp_get_wtime();                                                                /// Benchmarks model's evaluation
    for (sample = 0; sample < test_dim[0]; sample += 1)                                     /// Iterates through all examples of the evaluation dataset
    {
        zero_grad(fcn, Z, A, x_test[sample]);                                               /// Resets the neurons of the neural network
        feedforward(fcn, Z, A);                                                             /// Feeds forward the evaluation sample
        test_accuracy += model_accuracy(A[fcn->num_layers - 1], y_test[sample], fcn->sizes[fcn->num_layers - 1]);
                                                                                            /// Updates accuracy of the model based on the evaluation set
        test_loss += mse_loss(A[fcn->num_layers - 1], y_test[sample], fcn->sizes[fcn->num_layers - 1]);
                                                                                            /// Updates loss of the model based on the evaluation set
    }
    end = omp_get_wtime();                                                                  /// Terminates model's evaluation benchmark

    print_epoch_stats(-1, test_loss / MNIST_TEST, test_accuracy, end - start);              /// Prints evaluation loss, accuracy, and benchmark

    deallocate_Z(Z, fcn);                                                                   /// Deallocates `Z` off memory
    deallocate_A(A, fcn);                                                                   /// Deallocates `A` off memory

    return test_loss;
}
