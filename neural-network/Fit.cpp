
#include "Neural.hpp"
#include "Interface.hpp"

/**
 * Trains the given model. The model is a simple multi-
 * layer feed forward perceptron.
 *
 * @param[in] TRAIN the training dataset
 */
void nn::fit(dataset(&TRAIN))
{
    int shuffled_idx;                                                                       /// Decalres dample "pointer"
    double start, end;                                                                      /// Declares epoch benchmark checkpoints
    std::array<double, EPOCHS> loss;                                                        /// Declares container for training loss
    std::array<int, EPOCHS> validity;                                                       /// Declares container for training accuracy

    std::random_device rd;                                                                  /// Initializes non-deterministic random generator
    std::mt19937 gen(rd());                                                                 /// Seeds mersenne twister
    std::uniform_int_distribution<> dist(0, TRAIN.samples - 1);                             /// Distribute results between 0 and sample count exclusive
                                                                                            /// Change this depending on the ammount of loaded datasets
    for (int epoch = 0; epoch < EPOCHS; epoch += 1)                                         /// Trains model
    {
        loss[epoch] = 0.0;                                                                  /// Initializes epoch's training loss
        validity[epoch] = 0;                                                                /// Initializes epoch's training accuracy

        start = omp_get_wtime();                                                            /// Benchmarks epoch
        for (int sample = 0; sample < TRAIN.samples; sample += 1)                           /// Iterates through all examples of the training dataset
        {
            shuffled_idx = dist(gen);                                                       /// Selects a random example to avoid unshuffled dataset event
            zero_grad(TRAIN.X[shuffled_idx]);                                               /// Resets the neurons of the neural network
            forward();                                                                      /// Feeds forward the selected input
            back_propagation(TRAIN.Y[shuffled_idx]);                                        /// Computes the error for every neuron in the network
            optimize();                                                                     /// Optimizes weights using pack propagation
            loss[epoch] += mse_loss(TRAIN.Y[shuffled_idx], TRAIN.classes);                  /// Updates epoch's loss of the model
            validity[epoch] += accuracy(TRAIN.Y[shuffled_idx], TRAIN.classes);              /// Updates epoch's accuracy of the model
        }
        end = omp_get_wtime();                                                              /// Terminates epoch's benchmark

        loss[epoch] /= (TRAIN.samples + 0.0);                                               /// Averages epoch's loss of the model
        print_epoch_stats(epoch + 1, loss[epoch], validity[epoch], end - start);            /// Prints epoch's loss, accuracy and benchmark
    }
}

/**
 * Evaluates the given model.
 *
 * @param[in] TEST the evaluation dataset
 */

void nn::evaluate(dataset(&TEST))
{
    int validity = 0;
    double start, end, loss = 0.0;

    start = omp_get_wtime();                                                                /// Benchmarks model's evaluation
    for (int sample = 0; sample < TEST.samples; sample += 1)                                /// Iterates through all examples of the evaluation dataset
    {
        zero_grad(TEST.X[sample]);                                                          /// Resets the neurons of the neural network
        forward();                                                                          /// Feeds forward the evaluation sample
        loss += mse_loss(TEST.Y[sample], TEST.classes);                                     /// Updates loss of the model based on the evaluation set
        validity += accuracy(TEST.Y[sample], TEST.classes);                                 /// Updates accuracy of the model based on the evaluation set
    }
    end = omp_get_wtime();                                                                  /// Terminates model's evaluation benchmark

    loss /= (TEST.samples + 0.0);
    print_epoch_stats(-1, loss, validity, end - start);                                     /// Prints evaluation loss, accuracy, and benchmark
}
