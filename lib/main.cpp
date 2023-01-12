/*
____________________________________________________________________________
SOM From Scratch Main

_______

Overview: Train a SOM for a given keyed dataset (csv).

_______

Author(s): James D. Kern

Date: October 7, 2022

_______

Change Log:
To Do:
______________________________________________________________________________
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <regex>
#include "som.h"

using std::vector, std::string;

/**
 * @brief 
 * 
 * @return int 
 */
int main()
{   
    vector <string> labels;
    vector <vector <double>> X;
    std::ifstream  data("XXXXX.csv");
    const std::regex re{","};

    if (data.is_open())
    {
        std::string line;
        while(std::getline(data, line))
            {
                std::stringstream ss(line);
                std::vector <double> temp_vec;
                int i{0};
                while(std::getline(ss, line, ','))
                    {
                        if (i == 0)
                            {
                                ++i;
                                labels.push_back(line);
                            }
                        else
                        {
                            ++i;
                            temp_vec.push_back(std::stod(line));
                        }
                    }
                X.push_back(temp_vec);
            }
    }

    SOM TEST {12, 50000, 0.5, 0.5, "inverse", "gaussian", false, 10};
    TEST.train(X);
    TEST.output_results(X, labels, "output", false);

    return 0;
}


