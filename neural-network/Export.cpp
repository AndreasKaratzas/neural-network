
#include "Neural.hpp"

/**
 * Exports weights of neural network instance into a CSV file.
 *
 * @param[in] filename the CSV name
 */
void nn::export_weights(std::string filename)
{
    std::ofstream export_stream;										/// Defines an output file stream
    export_stream.open("./data/" + filename + ".csv");					/// Associates `export_stream` with a CSV file named after the `filename` variable
    for (int i = 1; i < layers.size() - 1; i += 1)						/// Loops through model's hidden layers
    {
        for (int j = 0; j < layers[i] - 1; j += 1)						/// Loops through layer's synapses
        {
            export_stream << "Neuron " << j << " Layer " << i << ",";
            for (int k = 0; k < layers[i - 1]; k += 1)
            {
                export_stream << weights[i - 1][j][k] << (j == layers[i] - 2 ? "" : ",");
            }															/// Export element of that array to the `export_stream` file stream
            export_stream << std::endl;
        }
        export_stream << std::endl;
    }

    for (int j = 0; j < layers[layers.size() - 1]; j += 1)				/// Loops through model's output layers
    {
        export_stream << "Neuron " << j << " Layer " << layers.size() - 1 << ",";
        for (int k = 0; k < layers[layers.size() - 1]; k += 1)
        {
            export_stream << weights[layers.size() - 2][j][k] << (j == layers[layers.size()] - 1 ? "" : ",");
        }																/// Export element of that array to the `export_stream` file stream
        export_stream << std::endl;
    }
    export_stream << std::endl;

    export_stream.close();												/// Closes file stream
}
