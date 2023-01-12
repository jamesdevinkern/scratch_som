/* 
____________________________________________________________________________
Self-organizing Map Class Definitions

_______

Overview: Definitions for implementation of SOM class.

_______

Author(s): James D. Kern

Date: October 6, 2022

_______

Change Log:
To Do:
______________________________________________________________________________
*/

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <iostream>
#include <fstream>
#include "som.h"
#include "neuron.h"

using std::vector, std::string, std::size_t, std::cout, std::pow, std::sqrt;

/**
 * @brief Construct a new SOM object (Default)
 * 
 * @param N The width of the Neuron grid space (will result in NxN square grid of neurons)
 * @param epochs Number of learning iterations
 * @param alpha Learning rate
 * @param sigma Neighborhood function search radius
 */
SOM::SOM(const unsigned int N, const unsigned int epochs, double alpha, double sigma)
    : N {N}, epochs {epochs}, alpha {alpha}, sigma {sigma}, learning_method {"inverse"}, neighborhood_function {"gaussian"}, seed {10}
{
    SOM::e_dbl = epochs;
}


/**
 * @brief Construct a new SOM object (alter learning _method, seed)
 * 
 * @param N The width of the Neuron grid space (will result in NxN square grid of neurons)
 * @param epochs Number of learning iterations
 * @param alpha Learning rate
 * @param sigma Neighborhood function search radius
 */
SOM::SOM(const unsigned int N, const unsigned int epochs, double alpha, double sigma, const string learning_method, const string neighborhood_function, bool record_quant=false, double seed=10)
    : N {N}, epochs {epochs}, alpha {alpha}, sigma {sigma}, learning_method {learning_method}, neighborhood_function {neighborhood_function}, record_quant {record_quant}, seed {seed} 
{
    SOM::e_dbl = epochs;
}


/**
 * @brief Intitializes the grid space of a SOM object
 * 
 * @param input_length Length of sample vectors
 * @returns void
 */
void SOM::initialize_map(size_t input_length)
{
    unsigned int N {SOM::N};
    vector <vector<Neuron>> temp_map(N);
    // vector <double> temp_weights(input_length); 
    std::mt19937 gen (SOM::seed); // Mersenne Twister pseudo-random generator
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    int cluster {0};
    
    for (size_t i{0}; i < N; ++i)
        for (size_t j{0}; j < N; ++j)
        {
            // Randomize Neuron's weight vector
            // TODO implement referencing to pointer on heap
            // vector <double> *temp_weights = new vector<double>(input_length);
            // vector <double> &weights_ref = *temp_weights;
            vector <double> temp_weights(input_length);
            std::generate(temp_weights.begin(), temp_weights.end(), [&]() { return unif(gen); });
            Neuron temp_neuron = Neuron(i, j, temp_weights);
            temp_neuron.cluster = cluster;
            cluster++;
            temp_map[i].push_back(temp_neuron);
            // delete temp_weights;
        }
            
    SOM::som_map = temp_map;
}


/**
 * @brief Determines the winning Neuron's position for a given input vector.
 * 
 * @param input_sample Input sample
 * @returns vector <int> position of winning Neuron for given input vector
 */
vector <int> SOM::winner(vector <double> input_sample)
{
    vector <vector <Neuron>>& som_map {SOM::som_map};
    double winner_d {0};
    vector <int> winner_pos {{0, 0}};

    for (int i{0}; i < SOM::N; ++i)
        for (int j{0}; j < SOM::N; ++j)
        {
            // Calculate the Euclidean distance between the
            // input vector and Neuron weight vector
            double temp_sum {0.};
            Neuron& temp_neuron {som_map[i][j]};

            for (size_t n{0}; n < temp_neuron.weights.size(); ++n)
                temp_sum += pow(input_sample[n] - temp_neuron.weights[n], 2);

            double euclidean_d= sqrt(temp_sum);

            if (i == 0)
            {
                winner_d = euclidean_d;
                winner_pos = {i, j};
            }
            else
            {
                if (euclidean_d < winner_d)
                {
                    winner_d = euclidean_d;
                    winner_pos = {i, j};
                }
            }
        }

    return winner_pos;
}


/**
 * @brief Update the weights for the winning Neuron and its neighborhood Neurons.
 * 
 * @param input_sample Input sample
 * @param winner_pos Coordinates of the winning Neuron
 */
void SOM::update_weights(vector <double> &input_sample, vector <int> &winner_pos)
{
    vector <vector <Neuron>>& som_map {SOM::som_map};
    
    for (int i{0}; i < SOM::N; ++i)
        for (int j{0}; j < SOM::N; ++j)
        {
            // Calculate the Euclidean distance between
            // Neuron_i_j and the winning Neuron
            double temp_sum {0.};
            Neuron& temp_neuron {som_map[i][j]};
            Neuron& winning_neuron {som_map[winner_pos[0]][winner_pos[1]]};

            for (size_t n{0}; n < temp_neuron.weights.size(); ++n)
                temp_sum += pow(winning_neuron.weights[n] - temp_neuron.weights[n], 2);

            double euclidean_d = sqrt(temp_sum);
            double lambda {};

            // Calculate lambda the influence the temp Neuron has in the neighborhood
            if (SOM::neighborhood_function == "bubble")
                lambda = 1.0;
            else if (SOM::neighborhood_function == "gaussian")
                lambda = exp(-pow(euclidean_d, 2) / (2 * pow(sigma, 2)));

            if (euclidean_d <= sigma)
                temp_neuron.update_weight(input_sample, lambda, alpha);
        }
}

/**
 * @brief Train a SOM object on a given vector of training samples.
 * 
 * @param X Vector of training samples
 */
void SOM::train(vector <vector <double>> &X)
{
    //TODO fix progress bar mess
    double progress {0.0};
    double X_iter {double(X.size())};
    double width {100.0};
    double alpha_init {alpha};
    double sigma_init {sigma};
    int quant_count {0};
    double vec_count {0.0};
    std::ofstream fileout("metric.txt");

    // Create vector of random ints for sample selection
    vector <int> rand_vec;
    for (size_t i{0}; i < X.size(); ++i)
        rand_vec.push_back(i);

    std::random_shuffle(rand_vec.begin(), rand_vec.end());

    fileout << "epoch" << ',' << "alpha" << ',' << "sigma" << "\n";

    // TODO add check for consistent input vector size in X

    // Initialize the SOM grid based on size of input vector size
    SOM::initialize_map(X[0].size());

    for (double i{0}; i <= epochs; ++i)
    {
        double i_iter {double(i)};
        double pos {width*progress};

        // Print progress bar to terminal
        std::cout << "|";
        for (int prog_i{0}; prog_i < width; ++prog_i)
        {
            if(prog_i < pos) std::cout << "=";
            else if (prog_i == ceil(pos)) std::cout << "*";
            else std::cout << " ";
        }
        std::cout << "| " << int(ceil(progress*100)) << " %\r";
        std::cout.flush();
        progress = {((i_iter + 1))/(epochs)};

        // Update SOM weights
        vector <double> temp_input {X[rand_vec[i]]};
        vector <int> x_winner { SOM::winner(temp_input) };
        SOM::update_weights(temp_input, x_winner);
    
        // Update the learning rate alpha after each iteration using the selected method
        if (learning_method == "inverse")
            alpha = alpha_init * (1 - (i + 1) / e_dbl); // Inverse time learning rate
        else if (learning_method == "linear")
            alpha = alpha_init * 1 / (i + 1); // Linear learning rate
        else if (learning_method == "power")
            alpha = alpha_init * exp(-(i + 1) / e_dbl); // Power learning rate

        // Update the neighborhood radius sigma after each iteration
            sigma = sigma_init * exp(-(i + 1) / e_dbl);

        fileout << i << ',' << alpha << ',' << sigma << "\n";
        // std::cout << "i: " << i+1 << " Epochs: " << e_dbl << " Alpha: " << SOM::alpha << " Sigma: " << SOM::sigma << std::endl;

        // Calculate the quantization error after each training iteration
        if (
            int(floor(i_iter / epochs * 100) / 100) % 1 == 0 && 
            quant_count == int(floor(i_iter / epochs * 100)) / 1 &&
            SOM::record_quant ||
            i_iter == epochs && SOM::record_quant
            )
            {
                double quant_sum {quantization_error(X)};
                SOM::quant_error.push_back(vector <double> {i_iter, quant_sum});
                quant_count += 1;
            }

        // Save the vectors associated with specifc cluster at each  
        // training iteration.
        // FOR PRESENTATION PURPOSES ONLY DELETE LATER
        // double vec_step {0.005};

        // if (
        //     vec_count == i_iter / epochs ||
        //     i_iter == epochs
        //     )
        //     {
        //         vector <vector <double>> hold_vec {};

        //         for (size_t k{0}; k < X.size(); ++k)
        //         {
        //             vector <double> input_sample {X[k]};
        //             int win_x {SOM::winner(input_sample)[0]};
        //             int win_y {SOM::winner(input_sample)[1]};
        //             int cluster {som_map[win_x][win_y].cluster};

        //             if (cluster == 1)
        //             {
        //                 hold_vec.push_back(input_sample);
        //             }
        //         }

        //         vector <double> mean_vec {};

        //         if (hold_vec.size() > 0)
        //         {
        //             for (size_t sample_i{0}; sample_i < X[0].size(); ++sample_i)
        //             {
        //                 double temp_sum {0};
        //                 for (size_t vec_i{0}; vec_i < hold_vec.size(); ++vec_i)
        //                 {
        //                     temp_sum += hold_vec[vec_i][sample_i];
        //                 }
        //                 mean_vec.push_back(temp_sum / hold_vec[0].size());
        //             }
                    
        //             mean_vec.push_back(i_iter);
        //             SOM::cluster_vec.push_back(mean_vec);

        //             vec_count += vec_step;
        //             vec_count = std::round(vec_count * 1000) / 1000;
        //         }
        //         else // to address instances of 0 samples for a given cluster/iter
        //         {
        //             for (size_t sample_i{0}; sample_i < X[0].size(); ++sample_i)
        //                 mean_vec.push_back(0);
                    
        //             mean_vec.push_back(i_iter);
        //             SOM::cluster_vec.push_back(mean_vec);

        //             vec_count += vec_step;
        //             vec_count = std::round(vec_count * 1000) / 1000;
        //         }
        //     }
            // FOR PRESENTATION PURPOSES ONLY DELETE LATER

        // Save the vectors associated with group 3 at each training iteration
        // FOR PRESENTATION PURPOSES ONLY DELETE LATER
        // double vec_step {0.005};

        // if (
        //     vec_count == i_iter / epochs ||
        //     i_iter == epochs
        //     )
        //     {
        //         vector <vector <double>> hold_vec {};

        //         for (size_t k{0}; k < som_map.size(); ++k)
        //         {
        //             for (size_t l{0}; l < som_map[k].size(); ++l)
        //             {
        //                 vector <double> temp_vec {som_map[k][l].weights};
        //                 temp_vec.push_back(som_map[k][l].cluster);
        //                 temp_vec.push_back(i_iter);
        //                 SOM::neuron_vec.push_back(temp_vec);
        //             }
        //         }

        //         vec_count += vec_step;
        //         vec_count = std::round(vec_count * 1000) / 1000;
        //     }
            // FOR PRESENTATION PURPOSES ONLY DELETE LATER

        // Reshuffle random vector if epochs exceeds length of input vector
        if (int(i_iter + 1) % X.size() == 0)
            {
                vector <int> temp_rand_vec;
                for (size_t k{0}; k < X.size(); ++k)
                    temp_rand_vec.push_back(k);
                std::random_shuffle(temp_rand_vec.begin(), temp_rand_vec.end());

                for (size_t k{0}; k < temp_rand_vec.size(); ++k)
                    rand_vec.push_back(temp_rand_vec[k]);
            }
    }
}

/**
 * @brief Displays the cluster associated with the winning Neuron for a given input sample.
 * 
 * @param input_sample Input sample
 */
void SOM::display_winner(vector <double> &input_sample)
{
    vector <int> winner {SOM::winner(input_sample)};
    int win_x {winner[0]};
    int win_y {winner[1]};
    int win_clust {som_map[win_x][win_y].cluster};

    std::cout << "{" << input_sample[0] << ", " << input_sample[1] << "} Cluster: " << win_clust << std::endl;
}

void SOM::output_results(vector <vector <double>> input_vector, vector <string> labels, string file_name, bool append_val)
{
    std::ofstream fileout(file_name + ".txt");
    std::ofstream quant_fileout(file_name + "_quant_error.txt");
    std::ofstream clust_fileout(file_name + "_clust_vec.txt");
    std::ofstream neuron_fileout(file_name + "_neuron_vec.txt");
    std::cout << "\nWriting output to file.\n";

    for (size_t i{0}; i < input_vector.size(); ++i)
    {
        vector <double> input_sample {input_vector[i]};
        string label {labels[i]};
        int win_x {SOM::winner(input_sample)[0]};
        int win_y {SOM::winner(input_sample)[1]};
        int cluster {som_map[win_x][win_y].cluster};

        fileout << label << ',';
        fileout << cluster << ',';
        
        // Option to include the sample's input vector in the output file
        if (append_val)
            for (size_t k{0}; k < input_sample.size(); ++k)
                fileout << input_sample[k] << ',';
        
        fileout << '\n';
    }

    // Save quantization error to file
    if (SOM::record_quant)
    {
        for (size_t i{0}; i < SOM::quant_error.size(); ++i)
        {
            double iter {SOM::quant_error[i][0]};
            double error {SOM::quant_error[i][1]};
            
            quant_fileout << iter << ',';
            quant_fileout << error << std::endl;
        }
    }
    

    // Save cluster vector to file
    // PRESENTATION ONLY DELETE LATER
    for (size_t i{0}; i < SOM::cluster_vec.size(); ++i)
    {
        vector <double> temp_vec {SOM::cluster_vec[i]};

        for (size_t k{0}; k < temp_vec.size(); ++k)
            clust_fileout << temp_vec[k] << ',';

        clust_fileout << std::endl;
    }
    // PRESENTATION ONLY DELETE LATER

    // Save neuron vector to file
    // PRESENTATION ONLY DELETE LATER
    for (size_t i{0}; i < SOM::neuron_vec.size(); ++i)
    {
        vector <double> temp_vec {SOM::neuron_vec[i]};

        for (size_t k{0}; k < temp_vec.size(); ++k)
            neuron_fileout << temp_vec[k] << ',';

        neuron_fileout << std::endl;
    }

    std::cout << "Done.\n";
}


//TODO
/**
 * @brief 
 * 
 * @param input_vector 
 * @return double Quantization error
 */
double SOM::quantization_error(const vector <vector <double>> &input_vector)
{
    double quant_sum {0};

    for (size_t i{0}; i < input_vector.size(); ++i)
    {
        double temp_sum {0};
        const vector <double> &input_sample {input_vector[i]};
        vector <int> winner {SOM::winner(input_sample)};
        Neuron& temp_neuron {som_map[winner[0]][winner[1]]};

        for (size_t n{0}; n < temp_neuron.weights.size(); ++n)
            temp_sum += input_sample[n] - temp_neuron.weights[n];

        quant_sum += abs(temp_sum);
    }

    return quant_sum;
}