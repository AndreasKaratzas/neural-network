
#include "Parser.h"

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
int parse_arguments(int argc, char* argv[], int* input_size, int* hidden_size, int* output_size, char* activation)
{
    int len, hidden_dim = 1;
    char* filename = argv[0];

    while ((argc > 1) && (argv[1][0] == '-'))                                           /// Loops through all arguments
    {
        switch (argv[1][1])                                                             /// Stops when there are no more arguments
        {
        case 'i':                                                                       /// '-i' option: This is used to give an input size for the first layer of the model
            *input_size = parse_integer(&argv[2][0]);
            break;
        case 'h':                                                                       /// '-h' option: This is used to give the size of a hidden layer of the model
            if (hidden_dim == array_sizeof(hidden_size) / sizeof(long double))
            {
                hidden_size = (int*)realloc(hidden_size, hidden_dim * sizeof(int));
                assert(hidden_size);
            }
            hidden_size[hidden_dim - 1] = parse_integer(&argv[2][0]);                   /// There can be more than one hidden layers, and all have to be initialized using the '-h' option
            hidden_dim += 1;
            break;
        case 'o':                                                                       /// '-o' option: This is used to give an output size for the last layer of the model
            *output_size = parse_integer(&argv[2][0]);
            break;
        case 'a':                                                                       /// '-a' option: This is used to give the method of filtering (activation) for each neuron
            len = strlen(&argv[2][0]);
            activation = (char*)realloc(activation, (len + 1) * sizeof(char));
            assert(activation);
            memcpy(activation, &argv[2][0], strlen(&argv[2][0]) + 1);
            break;
        default:
            usage(filename);                                                            /// If given option is invalid, the program prints the usage and terminates execution
        }
        argv += 2;
        argc -= 2;
    }

    hidden_dim -= 1;

    return hidden_dim;
}
