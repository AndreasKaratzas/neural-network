
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
    double start, end;
    std::vector<int> vec;

    nn fcn;                                                                                         /// Declares the image of the neural network
    dataset TRAIN;
    dataset TEST;

    setupConsole();
    hideCursor();
    clearScreen();

    parse_arguments(argc, argv, vec);                                                               /// Parses user arguments

    start = omp_get_wtime();                                                                        /// Initializes benchmark

    TRAIN.read_csv(TRAINING_DATA_FILEPATH, 0);
    TEST.read_csv(EVALUATION_DATA_FILEPATH, 1);

    fcn.compile(vec, -1.0, 1.0);                                                                    /// Initializes the neural network's image
    fcn.fit(TRAIN);                                                                                 /// Trains the model
    fcn.evaluate(TEST);                                                                             /// Evaluates the model

    end = omp_get_wtime();                                                                          /// Terminates the benchmark

    showCursor();

    std::cout << "\n\nBenchmark results: " << end - start << " seconds\n";                           /// Prints benchmark results

    return(0);
}
