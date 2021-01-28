
#include "Driver.hpp"

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
    int cli_rows, cli_cols, cursor_row, cursor_col;
    double start, end;
    std::vector<int> vec;

    nn fcn;                                                                                         /// Declares the image of the neural network
    dataset TRAIN(MNIST_CLASSES);                                                                   /// Declares training data subset
    dataset TEST(MNIST_CLASSES);                                                                    /// Declares evaluation data subset

    parse_arguments(argc, argv, vec);                                                               /// Parses user arguments
    start = omp_get_wtime();                                                                        /// Initializes benchmark

    TRAIN.read_csv(TRAINING_DATA_FILEPATH, 0);                                                      /// Initializes training data subset
    TEST.read_csv(EVALUATION_DATA_FILEPATH, 1);                                                     /// Initializes evaluation data subset

    fcn.compile(vec, -1.0, 1.0);                                                                    /// Initializes the neural network's image
    fcn.summary();                                                                                  /// Prints model structure
    fcn.fit(TRAIN);                                                                                 /// Trains the model
    fcn.evaluate(TEST);                                                                             /// Evaluates the model
    fcn.export_weights("mnist-fcn");

    end = omp_get_wtime();                                                                          /// Terminates the benchmark

    std::cout << "\n\nBenchmark results: " << end - start << " seconds\n";                          /// Prints benchmark results

    return(0);
}
