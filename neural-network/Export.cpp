
#include "Export.h"

/**
 * Exports a long double precision floating point vector into a CSV file.
 * The CSV will be stored in the project's directory.
 * 
 * @param[in] container the dynamic array (vector) to be exported
 * @param[in] filename the CSV name
 * @param[in] dim the size of the vector
 */
void export_double_vector(long double* (&container), char* filename, int dim)
{
    int row;

    FILE* stream;
    stream = fopen(filename, "w+");

    if (!stream)                                                /// Masks file parsing failure
    {
        printf("The file %s was not opened\n", filename);
    }
    else
    {
        for (row = 0; row < dim; row += 1)                      /// Iterates through the given vector
        {
            fprintf(stream, "%Lf\n", container[row]);           /// Saves cell value in one column
        }
        fclose(stream);
    }
}
