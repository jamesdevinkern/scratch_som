/*
____________________________________________________________________________
Self-organizing Map Class Header

_______

Overview: Header for implementation of SOM class.

_______

Author(s): James D. Kern

Date: October 6, 2022

_______

Change Log:
To Do:
______________________________________________________________________________
*/

#ifndef __SOM_H__
#define __SOM_H__

#include <vector>
#include "neuron.h"

using std::vector, std::string;

/**
 * @brief 
 * 
 */
class SOM
{
private:
    unsigned int N {}; // The width of the Neuron grid space (will result in NxN square grid of neurons)
    unsigned int epochs {}; double e_dbl {}; // Number of learning iterations, double needed for learning rate calculations
    double alpha {}; // Learning rate
    double sigma {}; // Neighborhood function search radius
    double seed {}; // Randomized function seed
    vector <vector <double>> quant_error {}; // Vector of quantization errors
    vector <vector <double>> cluster_vec {}; // Temp for presentation
    vector <vector <double>> neuron_vec {}; // Temp for presentation
    const string learning_method {""}; // Learning method (linear, inv_time, power)
    const string neighborhood_function {""}; // Neighborhood function (bubble, gaussian)
public:
    vector <vector <Neuron>> som_map; // Neuron grid space
    const bool record_quant {}; // Option to record quantization error

    SOM (
        unsigned int N, 
        unsigned int epochs, 
        double alpha, 
        double sigma
        ); // Default SOM Constructor

    SOM (
        unsigned int N, 
        unsigned int epochs, 
        double alpha, 
        double sigma, 
        const string learning_method,
        const string neighborhood_function,
        bool record_quant,
        double seed
        ); // SOM Constructor for chaning learning method and see

    void initialize_map(size_t input_length); // Initialize SOM grid
    vector <int> winner(vector <double> input_sample); // Determine sample's winning Neuron
    void update_weights(vector <double> &input_sample, vector <int> &winner_pos); // Update grid-space weights
    void train(vector <vector <double>> &X); // Train the SOM object for given vector of sample data
    void display_winner(vector <double> &input_sample); // Display the cluster of the associated winning Neuron
    void output_results(vector <vector <double>> input_vector, vector <string> labels, string file_name, bool append_val);
    double quantization_error(const vector <vector <double>> &input_vector);
};

#endif // __SOM_H__