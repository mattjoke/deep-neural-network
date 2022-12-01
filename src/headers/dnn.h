//
// Created by Matej Hakoš on 11/29/2022.
//

#ifndef DEEP_NEURAL_NETWORK_DNN_H
#define DEEP_NEURAL_NETWORK_DNN_H

#include <vector>

using namespace std;

struct ForwardPassOutput {
    vector<vector<double>> Z1;
    vector<vector<double>> A1;
    vector<vector<double>> Z2;
    vector<vector<double>> A2;
};

class dnn {
    static constexpr int INPUT_SIZE = 784;
    static constexpr int HIDDEN_SIZE = 20;
    static constexpr int OUTPUT_SIZE = 10;
    static constexpr int BATCH_SIZE = 100;

    vector<vector<double>> W1 = {};
    vector<double> b1 = {};
    vector<vector<double>> W2 = {};
    vector<double> b2 = {};
    double learning_rate = 0.01;

    // Momentum
    vector<vector<double>> vW1 = {};
    vector<double> vb1 = {};
    vector<vector<double>> vW2 = {};
    vector<double> vb2 = {};
    double beta = 0.1;

private:
    static int getIndexOfMaxValue(const vector<double> &input);

public:
    dnn();

    void init_weights_biases();

    ForwardPassOutput forward_propagation(const vector<vector<double>> &input);

    void backward_propagation(const vector<vector<double>> &input, const vector<vector<double>> &targets);

    static vector<vector<double>> ReLU(vector<vector<double>> &vector);

    static vector<vector<double>> derivative_ReLU(vector<vector<double>> &vector);

    static vector<vector<double>> softmax(const vector<vector<double>> &input);

    void
    gradient_descent(const vector<vector<double>> &input, const vector<vector<double>> &targets, int epochs);

    static double accuracy(const vector<vector<double>> &predicted, const vector<vector<double>> &ground_truth);

    vector<vector<double>> predict(const vector<vector<double>> &input);
};


#endif //DEEP_NEURAL_NETWORK_DNN_H
