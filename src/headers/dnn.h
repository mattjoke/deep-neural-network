//
// Created by Matej Hakoš on 11/29/2022.
//

#ifndef DEEP_NEURAL_NETWORK_DNN_H
#define DEEP_NEURAL_NETWORK_DNN_H

#include <vector>
#include "image_loader.h"

using namespace std;

struct ForwardPassOutput {
    vector<vector<double>> Z1;
    vector<vector<double>> A1;
    vector<vector<double>> Z2;
    vector<vector<double>> A2;
    vector<vector<double>> Z3;
    vector<vector<double>> A3;
};

class dnn {
    static constexpr int INPUT_SIZE = 784;
    static constexpr int HIDDEN_SIZE = 89;
    static constexpr int HIDDEN_SIZE2 = 30;
    static constexpr int OUTPUT_SIZE = 10;
    int BATCH_SIZE = 64; // batch size should be able to divide 10K, 60K and should be divisible by number of threads -> FIXED (any arbitrary number)

    vector<vector<double>> W1 = {};
    vector<double> b1 = {};
    vector<vector<double>> W2 = {};
    vector<double> b2 = {};
    vector<vector<double>> W3 = {};
    vector<double> b3 = {};
    double learning_rate = 0.01;

    // Momentum
    vector<vector<double>> vW1 = {};
    vector<double> vb1 = {};
    vector<vector<double>> vW2 = {};
    vector<double> vb2 = {};
    vector<vector<double>> vW3 = {};
    vector<double> vb3 = {};
    double beta = 0.9;

    // RMSProp
    vector<vector<double>> sW1 = {};
    vector<double> sb1 = {};
    vector<vector<double>> sW2 = {};
    vector<double> sb2 = {};
    vector<vector<double>> sW3 = {};
    vector<double> sb3 = {};
    double beta2 = 0.999;
    double epsilon = 1e-8;

    // Adam
    double eta = 1;

private:
    static int getIndexOfMaxValue(const vector<double> &input);

public:
    dnn();

    void setLR(double newLR) {
        learning_rate = newLR;
    }

    double getLR() const {
        return learning_rate;
    }

    void init_weights_biases();

    void forward_propagation(const vector<vector<double>> &input,
                             ForwardPassOutput &forward_pass_output);

    void backward_propagation(const vector<vector<double>> &input, const vector<vector<double>> &targets);

    static vector<vector<double>> ReLU(vector<vector<double>> &vector);

    static vector<vector<double>> derivative_ReLU(vector<vector<double>> &vector);

    static vector<vector<double>> softmax(const vector<vector<double>> &input);

    void
    gradient_descent(const vector<vector<double>> &input, const vector<vector<double>> &targets);

    static double accuracy(const vector<vector<double>> &predicted, const vector<vector<double>> &ground_truth);

    static double loss(const vector<vector<double>> &predicted, const vector<vector<double>> &ground_truth);

    vector<vector<double>> predict(const vector<vector<double>> &input);

    void train(image_loader *ih, size_t epochs);
};


#endif //DEEP_NEURAL_NETWORK_DNN_H
