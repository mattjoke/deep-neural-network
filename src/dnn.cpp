//
// Created by Matej Hakoš on 11/29/2022.
//

#include <random>
#include <iostream>
#include "headers/dnn.h"
#include "headers/math.h"

dnn::dnn() {
    this->W1 = vector<vector<double>>(784, vector<double>(20, 1));
    this->b1 = vector<double>(20);
    this->W2 = vector<vector<double>>(20, vector<double>(10, 0));
    this->b2 = vector<double>(10);
    this->init_weights_biases();
}

void dnn::init_weights_biases() {
    random_device rd;
    mt19937 gen(rd());

    // Initialize W1
    normal_distribution<> d1(0, 2 / sqrt(784));
    for (auto &row: this->W1) {
        for (auto &item: row) {
            item = d1(gen);
        }
    }
    // Initialize b1
    normal_distribution<> p1(0, 2 / sqrt(784));
    for (auto &item: this->b1) {
        item = p1(gen);
    }
    // Initialize W2
    normal_distribution<> d2(0, 2 / sqrt(20));
    for (auto &row: this->W2) {
        for (auto &item: row) {
            item = d2(gen);
        }
    }
    // Initialize b2
    normal_distribution<> p2(0, 2 / sqrt(20));
    for (auto &item: this->b2) {
        item = p2(gen);
    }

    this->vW1 = vector<vector<double>>(784, vector<double>(20, 0));
    this->vb1 = vector<double>(20, 0);
    this->vW2 = vector<vector<double>>(20, vector<double>(10, 0));
    this->vb2 = vector<double>(10, 0);
}


ForwardPassOutput dnn::forward_propagation(const vector<vector<double>> &input) {
    vector<vector<double>> Z1 = add(matmul(this->W1, input), this->b1);
    vector<vector<double>> A1 = ReLU(Z1);
    vector<vector<double>> Z2 = add(matmul(this->W2, A1), this->b2);
    vector<vector<double>> A2 = softmax(Z2);
    return {
            Z1,
            A1,
            Z2,
            A2
    };
}

void dnn::backward_propagation(const vector<vector<double>> &input, const vector<vector<double>> &targets) {
    if (input.size() != targets.size()) {
        throw invalid_argument("Backpropagation needs arguments with the same size!");
    }
    int output_size = targets.size();

    ForwardPassOutput forward_pass_output = forward_propagation(input);
    vector<vector<double>> Z1 = forward_pass_output.Z1;
    vector<vector<double>> A1 = forward_pass_output.A1;
    vector<vector<double>> Z2 = forward_pass_output.Z2;
    vector<vector<double>> A2 = forward_pass_output.A2;

    vector<vector<double>> dZ2 = subtract(A2, targets);
    vector<vector<double>> dW2 = multiply(1.0 / output_size, matmul(dZ2, transpose(A1)));
    vector<double> db2 = multiply_bias(1.0 / output_size, sum(dZ2));
    vector<vector<double>> dZ1 = multiply(derivative_ReLU(Z1), matmul(transpose(this->W2), dZ2));
    vector<vector<double>> dW1 = multiply(1.0 / output_size, matmul(dZ1, transpose(input)));
    vector<double> db1 = multiply_bias(1.0 / output_size, sum(dZ1));

    vector<vector<double>> newvW1 = add(multiply(this->beta, this->vW1), multiply(1 - this->beta, dW1));
    this->W1 = subtract(this->W1, multiply(this->learning_rate, newvW1));
    this->vW1 = newvW1;

    vector<double> newvb1 = add(multiply_bias(this->beta, this->vb1), multiply_bias(1 - this->beta, db1));
    this->b1 = subtract_bias(this->b1, multiply_bias(this->learning_rate, newvb1));
    this->vb1 = newvb1;

    vector<vector<double>> newvW2 = add(multiply(this->beta, this->vW2), multiply(1 - this->beta, dW2));
    this->W2 = subtract(this->W2, multiply(this->learning_rate, newvW2));
    this->vW2 = newvW2;

    vector<double> newvb2 = add(multiply_bias(this->beta, this->vb2), multiply_bias(1 - this->beta, db2));
    this->b2 = subtract_bias(this->b2, multiply_bias(this->learning_rate, newvb2));
    this->vb2 = newvb2;
}

vector<vector<double>> dnn::ReLU(vector<vector<double>> &vector) {
    for (auto &row: vector) {
        for (auto &item: row) {
            item = max(0.0, item);
        }
    }
    return vector;
}

vector<vector<double>> dnn::derivative_ReLU(vector<vector<double>> &vector) {
    for (auto &row: vector) {
        for (auto &item: row) {
            item = item > 0 ? 1 : 0;
        }
    }
    return vector;
}

vector<vector<double>> dnn::softmax(const vector<vector<double>> &input) {
    // Blame me for this
    vector<vector<double>> output;
    for (const auto &row: input) {
        vector<double> row_output;
        double sum = 0;
        for (const auto &item: row) {
            sum += exp(item);
        }
        for (const auto &item: row) {
            row_output.push_back(exp(item) / sum);
        }
        output.push_back(row_output);
    }
    return output;
}

void dnn::gradient_descent(const vector<vector<double>> &input, const vector<vector<double>> &targets,
                           int epochs) {
    int batch_size = 100;
    for (int i = 0; i < epochs; i++) {
        clock_t start = clock();
        for (size_t j = 0; j < input.size(); j += batch_size) {
            vector<vector<double>> mini_batch = {};
            vector<vector<double>> mini_batch_label = {};
            for (size_t k = j; k < j + batch_size; k++) {
                mini_batch.emplace_back(input[k]);
                mini_batch_label.emplace_back(targets[k]);
            }
            this->backward_propagation(mini_batch, mini_batch_label);
        }
//        this->backward_propagation(input, targets);
        clock_t end = clock();
        cout << "Epoch: " << i + 1 << endl;

        cout << "Last epoch took: " << double(end - start) / CLOCKS_PER_SEC << " s" << endl;
        // cout << "Accuracy: " << this->accuracy(predict(input), targets) << endl;
    }
}

int dnn::getIndexOfMaxValue(const vector<double> &input) {
    int index = 0;
    double max = 0;
    for (size_t i = 0; i < input.size(); i++) {
        if (input[i] > max) {
            max = input[i];
            index = i;
        }
    }
    return index;
}

double dnn::accuracy(const vector<vector<double>> &predicted, const vector<vector<double>> &ground_truth) {
    if (predicted.size() != ground_truth.size()) {
        throw invalid_argument("The predicted input should have the same size as the ground truth!");
    }
    int correct = 0;
    for (size_t i = 0; i < predicted.size(); i++) {
        int pr = getIndexOfMaxValue(predicted[i]);
        int gt = getIndexOfMaxValue(ground_truth[i]);
        if (pr == gt) {
            correct++;
        }
    }
    return (double) correct / ground_truth.size();
}

vector<vector<double>> dnn::predict(const vector<vector<double>> &input) {
    return forward_propagation(input).A2;
}
