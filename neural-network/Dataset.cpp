
#include "Dataset.hpp"

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
ssize_t dataset::getline(char** lineptr, size_t* n, FILE* stream)
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

void dataset::read_csv(const char* filename, int dataset_flag)
{
    FILE* stream;
    stream = fopen(filename, "r");

    int read = 0, columns = 0, rows = 1, current_width, intval, y_idx;
    char* line = NULL, delimiter[] = ",";
    std::string training_message = "Loading training dataset";
    std::string evalation_message = "Loading evaluation dataset";
    size_t len = 0;


    if (!stream)                                                                                                /// Mask failure in opening the given file
    {
        printf("The file %s was not opened\n", filename);
    }
    else
    {
        std::cout << "\n";

        progress_bar progress{ dataset_flag == 0 ? training_message : evalation_message, char(219), CLI_WINDOW_WIDTH };

        rewind(stream);

        X = new double* [(dataset_flag == 0 ? (int)MNIST_TRAIN : (int)MNIST_TEST)];                             /// Initializes the X instance, which is the container with the neural network's input samples
        Y = new double* [(dataset_flag == 0 ? (int)MNIST_TRAIN : (int)MNIST_TEST)];                             /// Initializes the Y instance, which is the container with the neural network's corresponding (expected) outputs

        if ((read = getline(&line, &len, stream)) != -1)                                                        /// Parse first row in the given CSV file
        {
            char* token = strtok(line, delimiter);                                                              /// Split row in tokens based on the file separator, which in this case is the comma character

            while (token != NULL)
            {
                token = strtok(NULL, delimiter);
                columns += 1;                                                                                   /// Count number of columns
            }
        }

        dimensions = columns - 1;

        while ((read = getline(&line, &len, stream)) != -1)                                                     /// Do the same for all the other rows
        {

            progress.indicate_progress((rows + 0.0) / (dataset_flag == 0 ? MNIST_TRAIN : MNIST_TEST), x, y);    /// Use a progress bar as user interface since the dataset might be large

            char* token = strtok(line, delimiter);

            columns = 0;

            X[rows - 1] = new double[dimensions];                                                               /// Initialize `X` placeholder. In this container, we will store the samples' input for the model
            Y[rows - 1] = new double[classes];                                                                  /// Initialize `Y` placeholder. In this container, we will store the samples' corresponding (expected) to train the model

            if (sscanf(token, "%d", &intval) != 1)                                                              /// Masks invalid data error
            {
                fprintf(stderr, "error - not an integer");                                                      /// Expecrting integer value type data
            }

            for (y_idx = 0; y_idx < classes; y_idx += 1)                                                        /// Convert integer to `Y` value for the model depending on the number of classes
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

            while (token != NULL)                                                                               /// Separate fetched line into input tokens using commas as separators
            {
                token = strtok(NULL, delimiter);
                if (token != NULL)
                {
                    if (sscanf(token, "%d", &intval) != 1)                                                      /// Mask string to number casting fault
                    {
                        fprintf(stderr, "error - not an integer");
                    }

                    X[rows - 1][columns] = (intval + 0.0) / 255.0;                                              /// Normalize data to avoid overflow. For the MNIST dataset, max cell value is 255
                    columns += 1;
                }
            }
            rows += 1;
        }
        fclose(stream);

        samples = rows - 1;

        x += 2;                                                                                                 /// Update CLI interface status
    }
}

/**
 * Fetches the class label. The `Y` placeholder an element initialized 
 * with 1.0 (one) indicating the sample's class. All other elements are
 * initialized with 0 (zero).
 * 
 * @param[in] sample the index of the sample in the dataset
 * 
 * @return an integer corresponding to the class of the given sample
 * 
 * @note    To solve double comparison fault due to precision representation error,
 *          we use the greater operand, since 0.0 < 0.9 and 1.0 > 0.9.
 */
int dataset::get_label(int sample)
{
    int label;

    for (int i = 0; i < classes; i += 1)
    {
        if (Y[sample][i] > 0.9)
        {
            label = i;
        }
    }

    return label;
}

/**
 * Prints every sample in the dataset.
 */
void dataset::print_dataset(void)
{
    for (int i = 0; i < samples; i += 1)
    {
        std::cout << "Sample " << i << " :";
        for (int j = 0; j < dimensions; j += 1)
        {
            std::cout << X[i][j] << " ";
            if (j % CLI_WINDOW_WIDTH == 0 && j != 0)                                                            /// If the number of elements in `X` placeholder is large, split the elements into multiple lines
            {
                std::cout << "\n";
            }
        }
        std::cout << "\tLabel: " << get_label(i);
        std::cout << std::endl;
    }
}
