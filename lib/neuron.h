/*
____________________________________________________________________________
Neuron Class Header

_______

Overview: Header for implementation of Neuron class.

_______

Author(s): James D. Kern

Date: October 6, 2022

_______

Change Log:
To Do:
______________________________________________________________________________
*/

#ifndef __NEURON_H__
#define __NEURON_H__

#include <vector>

using std::vector, std::string;

/**
 * @brief 
 * 
 */
class Neuron
{
public:
    vector <double> weights {};
    vector <int> position {};
    int cluster {};

    Neuron (int pos_x, int pos_y, vector <double> weights);
    void update_weight(
        vector <double> input_vector, 
        double lambda,
        double alpha
        ); // Update Neuron weight according to selected neighborhood function
};

#endif // __NEURON_H__