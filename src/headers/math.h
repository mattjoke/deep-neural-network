//
// Created by Matej Hakoš on 11/29/2022.
//

#ifndef DEEP_NEURAL_NETWORK_MATH_H
#define DEEP_NEURAL_NETWORK_MATH_H

#include <vector>
#include <cmath>

using namespace std;

vector<vector<double>> transpose(const vector<vector<double>> &input);

vector<vector<double>> matmul(const vector<vector<double>> &input1, const vector<vector<double>> &input2);

// Add functions
vector<vector<double>> add(const vector<vector<double>> &input1, const vector<double> &input2);

vector<vector<double>> add(const vector<vector<double>> &input1, const vector<vector<double>> &input2);

vector<double> add(const vector<double> &input1, const vector<double> &input2);

vector<vector<double>> add(const vector<vector<double>> &input1, double input2);

vector<double> add_bias(const vector<double> &input, double bias);

// Subtract functions
vector<vector<double>> subtract(const vector<vector<double>> &input1, const vector<vector<double>> &input2);

vector<double> subtract_bias(const vector<double> &input1, const vector<double> &input2);

// Multiply functions
vector<vector<double>> multiply(double input1, const vector<vector<double>> &input2);

vector<vector<double>> multiply(const vector<vector<double>> &input1, const vector<vector<double>> &input2);

vector<double> multiply_bias(double input1, const vector<double> &input2);

vector<double> multiply_bias(const vector<double> &input1, const vector<double> &input2);

// Divide functions
vector<vector<double>> divide(const vector<vector<double>> &input1, const vector<vector<double>> &input2);

vector<double> divide(const vector<double> &input1, const vector<double> &input2);

// Other functions
vector<double> sum(const vector<vector<double>> &input);

vector<vector<double>> sqrt(const vector<vector<double>> &input);

vector<double> sqrt(const vector<double> &input);


#endif //DEEP_NEURAL_NETWORK_MATH_H
