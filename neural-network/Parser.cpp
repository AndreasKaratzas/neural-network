
#include "Parser.hpp"

/**
 * Converts a string argument to integer.
 *
 * @param[in] argv the string to convert
 *
 * @return the integer that corresponds to that string
 *
 * @note    Visual studio has marked the `sscanf()` as deprecated routine.
 *          However, this is just for Windows OS.
 */
int parse_integer(char* argv)
{
    int intvar;

    if (sscanf(argv, "%d", &intvar) != 1)
    {
        fprintf(stderr, "error - not an integer");
    }

    return intvar;
}

/**
 * Parses all user arguments and initializes all necessary project attributes, such as the model's hyperparameters.
 *
 * @param[in] argc the number of user arguments
 * @param[in] argv the vector of the user arguments
 * @param[in] input_size the size of the input layer for the neural network
 * @param[in] hidden_size the vector with the number of neurons for each hidden layer in the neural network
 * @param[in] output_size the size of the output layer for the neural network
 * @param[in] activation string corresponding to the model's activation function
 *
 * @return the number of the hidden layers
 */
void parse_arguments(int argc, char* argv[], std::vector<int>& vec)
{
    int len, hidden_dim = 1;
    char* filename = argv[0];

    while ((argc > 1) && (argv[1][0] == '-'))                                           /// Loops through all arguments
    {
        switch (argv[1][1])                                                             /// Stops when there are no more arguments
        {
        case 'i':                                                                       /// '-i' option: This is used to give an input size for the first layer of the model
            vec.push_back(parse_integer(&argv[2][0]));
            break;
        case 'h':                                                                       /// '-h' option: This is used to give the size of a hidden layer of the model
            vec.push_back(parse_integer(&argv[2][0]));                                  /// There can be more than one hidden layers, and all have to be initialized using the '-h' option
            break;
        case 'o':                                                                       /// '-o' option: This is used to give an output size for the last layer of the model
            vec.push_back(parse_integer(&argv[2][0]));
            break;
        default:
            usage(filename);                                                            /// If given option is invalid, the program prints the usage and terminates execution
        }
        argv += 2;
        argc -= 2;
    }
}
