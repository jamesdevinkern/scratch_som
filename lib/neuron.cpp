/* 
____________________________________________________________________________
Neuron Class Definitions

_______

Overview: Definitions for implementation of Neuron class.

_______

Author(s): James D. Kern

Date: October 6, 2022

_______

Change Log:
To Do:
______________________________________________________________________________
*/

#include <vector>
#include <iostream>
#include <cmath>
#include "neuron.h"

using std::vector, std::size_t, std::pow;

/**
 * @brief Construct a new Neuron:: Neuron object
 * 
 * @param pos_x 
 * @param pos_y 
 * @param weights 
 */
Neuron::Neuron(int pos_x, int pos_y, vector <double> weights)
    : position {pos_x, pos_y}, weights {weights}
{

}

/**
 * @brief 
 * 
 * @param input_vector 
 * @param alpha 
 */
void Neuron::update_weight(
    vector <double> input_vector, 
    double lambda,
    double alpha 
    )
    {
        // std::cout << "Neuron: (" << Neuron::position[0] << ", " << Neuron::position[1] << ") lambda: " << lambda << std::endl;
        for (size_t i{0}; i < weights.size(); ++i)
        {
            // std::cout << "Old: "<< weights[i];
            weights[i] = weights[i] + alpha * lambda * (input_vector[i] - weights[i]);
            // std::cout << " New: " << weights[i] << std::endl;
        }

        return;
    }


