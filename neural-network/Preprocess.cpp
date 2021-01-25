
#include "Preprocess.h"

/**
 * Processes a line of a given file stream.
 * 
 * @param[in] lineptr a vector of strings
 * @param[in, out] n length of a given row
 * @param[in] stream the file stream to parse
 * 
 * @return integer type that is used to mask failure during read operation
 * 
 * @remark  The original code is public domain -- Will Hartung 4/9/09
 *          Modifications, public domain as well, by Antti Haapala, 11/10/17
 *          Switched to getc on 5/23/19
 *          https://stackoverflow.com/questions/735126/are-there-alternate-implementations-of-gnu-getline-interface/
 */
ssize_t getline(char** lineptr, size_t* n, FILE* stream)
{
    size_t read;
    int character;

    if (lineptr == NULL || stream == NULL || n == NULL)
    {
        errno = EINVAL;
        return -1;
    }

    character = getc(stream);

    if (character == EOF)
    {
        return -1;
    }

    if (*lineptr == NULL)
    {
        *lineptr = (char*)malloc(128);

        if (*lineptr == NULL)
        {
            return -1;
        }

        *n = 128;
    }

    read = 0;

    while (character != EOF)
    {
        if (read + 1 >= *n)
        {
            size_t new_size = *n + (*n >> 2);

            if (new_size < 128)
            {
                new_size = 128;
            }

            char* new_ptr = (char*)realloc(*lineptr, new_size);

            if (new_ptr == NULL)
            {
                return -1;
            }
            *n = new_size;
            *lineptr = new_ptr;
        }

        ((unsigned char*)(*lineptr))[read++] = character;

        if (character == '\n')
        {
            break;
        }

        character = getc(stream);
    }

    (*lineptr)[read] = '\0';
    return read;
}

/**
 * Parses a CSV file and splits contents into input and output, to train a neural network.
 * 
 * @param[in] filename the file path of the CSV file to parse
 * @param[in, out] X the container where any input for the neural network is stored
 * @param[in, out] Y the container where the corresponding (expected) output for the neural network is stored
 * @param[in, out] dimensions the number of rows and columns successfully parsed
 * @param[in] dataset_flag  if `0`, then the function parses training data
 *                          if `1`, then the function parses evaluation data
 * 
 * @note The structure of the csv must obey the following rules:
 *          * The first row contains the column descriptions
 *          * All other cells contain integers
 *          * The first column is the expected output
 *          * All other columns operate as input for the neural network
 * 
 * @note    The `dataset_flag` is nothing more than an indicator for the UI. In
 *          other words it only tells the function which message to display, the
 *          message about parsing either the training or the evaluation data subset.
 */
void read_csv(const char* filename, long double** (&X), long double** (&Y), int* (&dimensions), int dataset_flag)
{
    FILE* stream;
    stream = fopen(filename, "r");

    int read = 0, columns = 0, rows = 1, current_width, intval, y_idx;
    char* line = NULL, overlay[CLI_WINDOW_WIDTH], delimiter[] = ",", training_message[] = "Loading training dataset", evalation_message[] = "Loading evaluation dataset";
    size_t len = 0;

    for (current_width = 0; current_width < CLI_WINDOW_WIDTH; current_width += 1)                   /// Initializes progress bar overlay
    {
        overlay[current_width] = '|';
    }

    while (NULL == (Y = (long double**)malloc(1 * sizeof(long double*)))) {}                        /// Initializes the X instance, which is the container with the neural network's input samples
    while (NULL == (X = (long double**)malloc(1 * sizeof(long double*)))) {}                        /// Initializes the Y instance, which is the container with the neural network's corresponding (expected) outputs
    
    if (!stream)
    {
        printf("The file %s was not opened\n", filename);                                           /// Mask failure in opening the given file
    }
    else
    {
        printf("\n");

        rewind(stream);

        if ((read = getline(&line, &len, stream)) != -1)                                            /// Parse first row in the given CSV file
        {
            char* token = strtok(line, delimiter);                                                  /// Split row in tokens based on the file separator, which in this case is the comma character

            while (token != NULL)
            {
                token = strtok(NULL, delimiter);
                columns += 1;                                                                       /// Count number of columns
            }
        }

        const int N_SAMPLES = columns - 1;

        dimensions[1] = N_SAMPLES;

        while ((read = getline(&line, &len, stream)) != -1)                                         /// Do the same for all the other rows
        {
            printProgress((rows + 0.0) / (dataset_flag == 0 ? MNIST_TRAIN : MNIST_TEST), CLI_WINDOW_WIDTH, overlay, (dataset_flag == 0 ? training_message : evalation_message));
                                                                                                    /// Use a aprogress bar as user interface since the dataset might be large
            char* token = strtok(line, delimiter);

            columns = 0;

            if (rows > sizeof X / sizeof X[0])                                                      /// Reallocate more memory space
            {
                while (NULL == (X = (long double**)realloc(X, rows * sizeof(long double*)))) {}
                while (NULL == (Y = (long double**)realloc(Y, rows * sizeof(long double*)))) {}
            }

            while (NULL == (X[rows - 1] = (long double*)malloc(N_SAMPLES * sizeof(long double)))) {}
            while (NULL == (Y[rows - 1] = (long double*)malloc(MNIST_CLASSES * sizeof(long double)))) {}

            if (sscanf(token, "%d", &intval) != 1)                                                  /// Mask string to number casting fault
            {
                fprintf(stderr, "error - not an integer");
            }

            for (y_idx = 0; y_idx < MNIST_CLASSES; y_idx += 1)                                      /// Convert expected output to neural network output depending on the dataset's classes count
            {
                if (y_idx == intval)
                {
                    Y[rows - 1][y_idx] = 1.0;
                }
                else
                {
                    Y[rows - 1][y_idx] = 0.0;
                }
            }

            while (token != NULL)
            {
                token = strtok(NULL, delimiter);
                if (token != NULL)
                {
                    if (sscanf(token, "%d", &intval) != 1)
                    {
                        fprintf(stderr, "error - not an integer");
                    }

                    X[rows - 1][columns] = (intval + 0.0) / 255.0;                                  /// Normalize data to avoid overflow. For the MNIST dataset, max cell value is 255

                    columns += 1;
                }
            }
            rows += 1;
        }
        fclose(stream);

        dimensions[0] = rows - 1;
    }
}

/**
 * Drives training and evaluation dataset parsing.
 * 
 * @param[in, out] x_train the container where any input for the neural network is stored regarding the training data subset
 * @param[in, out] y_train the container where the corresponding (expected) output for the neural network is stored regarding the training data subset
 * @param[in, out] train_dim the number of rows and columns successfully parsed regarding the training data subset
 * @param[in, out] x_test the container where any input for the neural network is stored regarding the evaluation data subset
 * @param[in, out] y_test the container where the corresponding (expected) output for the neural network is stored regarding the evaluation data subset
 * @param[in, out] test_dim the number of rows and columns successfully parsed regarding the evaluation data subset
 */
void train_test_split(long double** (&x_train), long double** (&y_train), int* (&train_dim), long double** (&x_test), long double** (&y_test), int* (&test_dim))
{
    read_csv(TRAINING_DATA_FILEPATH, x_train, y_train, train_dim, 0);
    read_csv(EVALUATION_DATA_FILEPATH, x_test, y_test, test_dim, 1);
    printf("\n");

    assert(train_dim[1] == test_dim[1]);                                                            /// Mask compatibility failure in dataset
}
