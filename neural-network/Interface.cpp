
#include "Interface.h"

/**
 * Prints information regarding the usage and the available options of the project.
 * 
 * @param[in] filename the filepath of the executable
 * 
 * @note Upon this function's execution, the program is terminated.
 */
void usage(char* filename)
{
    printf("Usage of %s:\n", filename);
    printf("\t:option \'-i\': integer \t - \t The size of the input layer for the neural network.\n");
    printf("\t:option \'-h\': integer \t - \t The size of a hidden layer for the neural network.\n\t\t\t\t\t There can be multiple hidden layers. For every hidden layer, use this option.\n");
    printf("\t:option \'-o\': integer \t - \t The size of the output layer for the neural network.\n");
    printf("\t:option \'-a\': string \t - \t The activation function for the neural network.\n\t\t\t\t\t Possible option(s): \"relu\" and \"sigmoid\".\n");
    exit(8);
}

/**
 * Prints the contents of a matrix. More specifically, it prints the contents
 * of a 2D long double precision floating point dynamic array.
 * 
 * @param[in] container the 2D long double precision floating point dynamic array
 * @param[in] num_layers the number of rows of the matrix
 * @param[in] sizes the column dictionary for each row
 * @param[in] container_typename a symbolic variable name for the printed container
 */
void print_2D_DOUBLE_vector(long double** container, int num_layers, int* sizes, char* container_typename)
{
    int i, j, level = 0;

    printf("For \"%s\":\n", container_typename);
    for (i = 0; i < num_layers; i += 1)
    {
        printf("\tLevel %d\n\t\t", level++);
        for (j = 0; j < sizes[i]; j += 1)
        {
            printf("%Lf ", container[i][j]);
            if (j % CLI_WINDOW_WIDTH == 0 && j != 0)                        /// Uses the window length defined in `Common.h`
            {
                printf("\n\t");
            }
        }
        printf("\n");
    }
    printf("\n");
}

/**
 * Prints the `delta` variable used during back propagation.
 * 
 * @param[in] delta the 2D long double precision floating point dynamic array
 * @param[in] num_layers the number of rows of the matrix
 * @param[in] sizes the column dictionary for each row
 * @param[in] container_typename a symbolic variable name for the printed container
 */
void print_delta(long double** delta, int num_layers, int* sizes, char* container_typename)
{
    int i, j, level = 0;

    printf("For \"%s\":\n", container_typename);
    for (i = 0; i < num_layers - 1; i += 1)
    {
        printf("\tLevel %d\n\t\t", level++);
        for (j = 0; j < sizes[i + 1]; j += 1)
        {
            printf("%Lf ", delta[i][j]);
            if (j % CLI_WINDOW_WIDTH == 0 && j != 0)
            {
                printf("\n\t");
            }
        }
        printf("\n");
    }
    printf("\n");
}

/**
 * Prints the contents of a tensor. More specifically, it prints the contents
 * of a 3D long double precision floating point dynamic array.
 * 
 * @param[in] container the 3D long double precision floating point dynamic array
 * @param[in] num_layers the number of rows of the tensor
 * @param[in] sizes the column dictionary for each row
 * @param[in] container_typename a symbolic variable name for the printed container
 */
void print_3D_DOUBLE_vector(long double*** container, int num_layers, int* sizes, char* container_typename)
{
    int i, j, k, level = 0, sub_level;

    for (i = 0; i < num_layers - 2; i += 1)
    {
        sub_level = 0;
        printf("\tLevel %d\n", level++);
        for (j = 0; j < sizes[i + 1] - 1; j += 1)
        {
            printf("\t\tSub-Level %d\n\t\t\t", sub_level++);
            for (k = 0; k < sizes[i]; k += 1)
            {
                printf("%Lf ", container[i][j][k]);
                if (k % CLI_WINDOW_WIDTH == 0 && k != 0)
                {
                    printf("\n\t");
                }
            }
            printf("\n");
        }
        printf("\n");
    }

    sub_level = 0;

    printf("\tLevel %d\n", level++);
    for (j = 0; j < sizes[num_layers - 1]; j += 1)
    {
        printf("\t\tSub-Level %d\n\t\t\t", sub_level++);
        for (k = 0; k < sizes[num_layers - 2]; k += 1)
        {
            printf("%Lf ", container[i][j][k]);
            if (k % CLI_WINDOW_WIDTH == 0 && k != 0)
            {
                printf("\n\t");
            }
        }
        printf("\n");
    }
    printf("\n");
}

/**
 * Prints the contents of a matrix. More specifically, it prints the contents
 * of a 2D dynamic integer array.
 *
 * @param[in] container the 2D dynamic integer array
 * @param[in] container_typename a symbolic variable name for the printed container
 * @param[in] dim_1 the number of rows of the matrix
 * @param[in] dim_2 the number of columns of the matrix
 */
void print_2D_INT_array(int** container, char* container_typename, int dim_1, int dim_2)
{
    int i, j;

    printf("For \"%s\":\n", container_typename);
    for (i = 0; i < dim_1; i += 1)
    {
        for (j = 0; j < dim_2; j += 1)
        {
            printf("%d ", container[i][j]);
            if (j % CLI_WINDOW_WIDTH == 0 && j != 0)
            {
                printf("\n\t");
            }
        }
        printf("\n");
    }
    printf("\n");
}

/**
 * Prints the contents of a matrix. More specifically, it prints the contents
 * of a 2D dynamic integer array.
 *
 * @param[in] container the 2D dynamic integer array
 * @param[in] container_typename a symbolic variable name for the printed container
 * @param[in] dim_1 the number of rows of the matrix
 * @param[in] dim_2 the number of columns of the matrix
 */
void print_1D_INT_array(int* container, char* container_typename, int dim)
{
    int i;

    printf("%s:\t", container_typename);
    for (i = 0; i < dim; i += 1)
    {
        printf("%d ", container[i]);
        if (i % CLI_WINDOW_WIDTH == 0 && i != 0)
        {
            printf("\n\t");
        }
    }
    printf("\n");
}

/**
 * Prints epoch stats. More specifically, it prints the epoch's number 
 * along with the model's acuracy and loss. It also prints the epoch's benchmark.
 * 
 * @param[in] epoch the epoch's number
 * @param[in] epoch_loss the model's loss during a certain epoch of training or evaluation
 * @param[in] epoch_accuracy the model's accuracy during a certain epoch of training or evaluation
 * @param[in] benchmark the epoch's benchmark
 */
void print_epoch_stats(int epoch, long double epoch_loss, int epoch_accuracy, double benchmark)
{
    if (epoch == -1)
    {
        printf("\n\n[EVALUATION] [LOSS %.5e] [ACCURACY %6d out of %6d] Work took %3d seconds", epoch_loss, epoch_accuracy, (int)MNIST_TEST, (int)benchmark);
    }
    else
    {
        printf("\n[EPOCH %4d] [LOSS %.5e] [ACCURACY %6d out of %6d] Work took %3d seconds", epoch, epoch_loss, epoch_accuracy, (int)MNIST_TRAIN, (int)benchmark);
    }
}

/**
 * Prints the executable's arguments.
 * 
 * @param[in] input_size the size of the model's input layer
 * @param[in] hidden_dim the sizes of the model's hidden layers
 * @param[in] output size the size of the model's output layer
 * @param[in] activation the model's actuvation function name
 */
void print_parser_results(int input_size, int* hidden_size, int hidden_dim, int output_size, char* activation)
{
    printf("Input Layer:\t\t%d\n", input_size);
    print_1D_INT_array(hidden_size, "Hidden Layer(s)", hidden_dim);
    printf("Output layer:\t\t%d\n", output_size);
    printf("\nActivation function:\t%s\n", activation);
}
