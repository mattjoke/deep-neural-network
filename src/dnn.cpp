//
// Created by Matej Hakoš on 11/29/2022.
//

#include <random>
#include <iostream>
#include <functional>
#include <thread>
#include "headers/dnn.h"
#include "headers/math.h"

dnn::dnn() {
    this->W1 = vector<vector<double>>(INPUT_SIZE, vector<double>(HIDDEN_SIZE, 0));
    this->b1 = vector<double>(HIDDEN_SIZE);
    this->W2 = vector<vector<double>>(HIDDEN_SIZE, vector<double>(HIDDEN_SIZE2, 0));
    this->b2 = vector<double>(HIDDEN_SIZE2);

    // W3
    this->W3 = vector<vector<double>>(HIDDEN_SIZE2, vector<double>(OUTPUT_SIZE, 0));
    this->b3 = vector<double>(OUTPUT_SIZE);

    this->init_weights_biases();
}

void dnn::init(image_loader *ih, size_t epochs) {
    for (size_t i = 0; i < epochs; i++) {
        cout << "Epoch " << i + 1 << endl;
        this->gradient_descent(ih->get_all_images(), ih->get_all_labels(), 1);
        vector<vector<double>> predictions = this->predict(ih->get_all_images());
        double total_loss = loss(predictions, ih->get_all_labels());
        cout << "Loss for this epoch: " << total_loss << endl;
        ih->shuffle();
    }
}

void dnn::init_weights_biases() {
    random_device rd;
    mt19937 gen(rd());

    // Initialize W1
    normal_distribution<> d1(0, 2 / sqrt(INPUT_SIZE));
    for (auto &row: this->W1) {
        for (auto &item: row) {
            item = d1(gen);
        }
    }
    // Initialize b1
    normal_distribution<> p1(0, 2 / sqrt(INPUT_SIZE));
    for (auto &item: this->b1) {
        item = p1(gen);
    }
    // Initialize W2
    normal_distribution<> d2(0, 2 / sqrt(HIDDEN_SIZE));
    for (auto &row: this->W2) {
        for (auto &item: row) {
            item = d2(gen);
        }
    }
    // Initialize b2
    normal_distribution<> p2(0, 2 / sqrt(HIDDEN_SIZE));
    for (auto &item: this->b2) {
        item = p2(gen);
    }

    // Initialize W3
    normal_distribution<> d3(0, 2 / sqrt(HIDDEN_SIZE2));
    for (auto &row: this->W3) {
        for (auto &item: row) {
            item = d3(gen);
        }
    }
    // Initialize b3
    normal_distribution<> p3(0, 2 / sqrt(HIDDEN_SIZE2));
    for (auto &item: this->b3) {
        item = p3(gen);
    }

    // Momentum
    this->vW1 = vector<vector<double>>(INPUT_SIZE, vector<double>(HIDDEN_SIZE, 0));
    this->vb1 = vector<double>(HIDDEN_SIZE, 0);
    this->vW2 = vector<vector<double>>(HIDDEN_SIZE, vector<double>(HIDDEN_SIZE2, 0));
    this->vb2 = vector<double>(HIDDEN_SIZE2, 0);
    this->vW3 = vector<vector<double>>(HIDDEN_SIZE2, vector<double>(OUTPUT_SIZE, 0));
    this->vb3 = vector<double>(OUTPUT_SIZE, 0);

    // RMSProp
    this->sW1 = vector<vector<double>>(INPUT_SIZE, vector<double>(HIDDEN_SIZE, 0));
    this->sb1 = vector<double>(HIDDEN_SIZE, 0);
    this->sW2 = vector<vector<double>>(HIDDEN_SIZE, vector<double>(HIDDEN_SIZE2, 0));
    this->sb2 = vector<double>(HIDDEN_SIZE2, 0);
    this->sW3 = vector<vector<double>>(HIDDEN_SIZE2, vector<double>(OUTPUT_SIZE, 0));
    this->sb3 = vector<double>(OUTPUT_SIZE, 0);
}


void dnn::forward_propagation(const vector<vector<double>> &input, ForwardPassOutput &forward_pass_output) {
    forward_pass_output.Z1 = add(matmul(this->W1, input), this->b1);
    forward_pass_output.A1 = ReLU(forward_pass_output.Z1);
    forward_pass_output.Z2 = add(matmul(this->W2, forward_pass_output.A1), this->b2);
    forward_pass_output.A2 = ReLU(forward_pass_output.Z2);
    forward_pass_output.Z3 = add(matmul(this->W3, forward_pass_output.A2), this->b3);
    forward_pass_output.A3 = softmax(forward_pass_output.Z3);
}

// input - batch of images
void dnn::backward_propagation(const vector<vector<double>> &input, const vector<vector<double>> &targets) {
    if (input.size() != targets.size()) {
        throw invalid_argument("Backpropagation needs arguments with the same size!");
    }
    int output_size = targets.size();
//    vector<vector<double>> in1 = {};
//    vector<vector<double>> in2 = {};
//    vector<vector<double>> in3 = {};
//    vector<vector<double>> in4 = {};
//    ForwardPassOutput forward_pass_output1;
//    ForwardPassOutput forward_pass_output2;
//    ForwardPassOutput forward_pass_output3;
//    ForwardPassOutput forward_pass_output4;
//
//    for(size_t i = 0; i < input.size(); i++) {
//        switch (i % 4) {
//            case 0: in1.emplace_back(input[i]);
//            break;
//            case 1: in2.emplace_back(input[i]);
//            break;
//            case 2: in3.emplace_back(input[i]);
//            break;
//            case 3: in4.emplace_back(input[i]);
//            break;
//            default: in1.emplace_back(input[i]);
//            break;
//        }
//    }
//
//    auto thread1 = std::thread(&dnn::forward_propagation, this, std::ref(in1), std::ref(forward_pass_output1));
//    auto thread2 = std::thread(&dnn::forward_propagation, this, std::ref(in2), std::ref(forward_pass_output2));
//    auto thread3 = std::thread(&dnn::forward_propagation, this, std::ref(in3), std::ref(forward_pass_output3));
//    auto thread4 = std::thread(&dnn::forward_propagation, this, std::ref(in4), std::ref(forward_pass_output4));
//    thread1.join();
//    thread2.join();
//    thread3.join();
//    thread4.join();
//
//    ForwardPassOutput forward_pass_output;
//    for (size_t i=0; i < forward_pass_output1.Z1.size(); i++) {
//        forward_pass_output.Z1.emplace_back(forward_pass_output1.Z1[i]);
//        forward_pass_output.Z1.emplace_back(forward_pass_output2.Z1[i]);
//        forward_pass_output.Z1.emplace_back(forward_pass_output3.Z1[i]);
//        forward_pass_output.Z1.emplace_back(forward_pass_output4.Z1[i]);
//    }
//    for (size_t i=0; i < forward_pass_output1.A1.size(); i++) {
//        forward_pass_output.A1.emplace_back(forward_pass_output1.A1[i]);
//        forward_pass_output.A1.emplace_back(forward_pass_output2.A1[i]);
//        forward_pass_output.A1.emplace_back(forward_pass_output3.A1[i]);
//        forward_pass_output.A1.emplace_back(forward_pass_output4.A1[i]);
//    }
//    for (size_t i=0; i < forward_pass_output1.Z2.size(); i++) {
//        forward_pass_output.Z2.emplace_back(forward_pass_output1.Z2[i]);
//        forward_pass_output.Z2.emplace_back(forward_pass_output2.Z2[i]);
//        forward_pass_output.Z2.emplace_back(forward_pass_output3.Z2[i]);
//        forward_pass_output.Z2.emplace_back(forward_pass_output4.Z2[i]);
//    }
//    for (size_t i=0; i < forward_pass_output1.A2.size(); i++) {
//        forward_pass_output.A2.emplace_back(forward_pass_output1.A2[i]);
//        forward_pass_output.A2.emplace_back(forward_pass_output2.A2[i]);
//        forward_pass_output.A2.emplace_back(forward_pass_output3.A2[i]);
//        forward_pass_output.A2.emplace_back(forward_pass_output4.A2[i]);
//    }
//
//    for (size_t i=0; i < forward_pass_output1.Z3.size(); i++) {
//        forward_pass_output.Z3.emplace_back(forward_pass_output1.Z3[i]);
//        forward_pass_output.Z3.emplace_back(forward_pass_output2.Z3[i]);
//        forward_pass_output.Z3.emplace_back(forward_pass_output3.Z3[i]);
//        forward_pass_output.Z3.emplace_back(forward_pass_output4.Z3[i]);
//    }
//
//    for (size_t i=0; i < forward_pass_output1.A3.size(); i++) {
//        forward_pass_output.A3.emplace_back(forward_pass_output1.A3[i]);
//        forward_pass_output.A3.emplace_back(forward_pass_output2.A3[i]);
//        forward_pass_output.A3.emplace_back(forward_pass_output3.A3[i]);
//        forward_pass_output.A3.emplace_back(forward_pass_output4.A3[i]);
//    }

    ForwardPassOutput forward_pass_output;
    forward_propagation(input, forward_pass_output);
    vector<vector<double>> Z1 = forward_pass_output.Z1;
    vector<vector<double>> A1 = forward_pass_output.A1;
    vector<vector<double>> Z2 = forward_pass_output.Z2;
    vector<vector<double>> A2 = forward_pass_output.A2;
    vector<vector<double>> Z3 = forward_pass_output.Z3;
    vector<vector<double>> A3 = forward_pass_output.A3;


    vector<vector<double>> dZ3 = subtract(A3, targets);
    vector<vector<double>> dW3 = multiply(1.0 / output_size, matmul(dZ3, transpose(A2)));
    vector<double> db3 = multiply_bias(1.0 / output_size, sum(dZ3));

    vector<vector<double>> dZ2 = multiply(derivative_ReLU(Z2), matmul(transpose(this->W3), dZ3));
    vector<vector<double>> dW2 = multiply(1.0 / output_size, matmul(dZ2, transpose(A1)));
    vector<double> db2 = multiply_bias(1.0 / output_size, sum(dZ2));

    vector<vector<double>> dZ1 = multiply(derivative_ReLU(Z1), matmul(transpose(this->W2), dZ2));
    vector<vector<double>> dW1 = multiply(1.0 / output_size, matmul(dZ1, transpose(input)));
    vector<double> db1 = multiply_bias(1.0 / output_size, sum(dZ1));


    // Use momentum to update weights and biases
    vector<vector<double>> newvW1 = add(multiply(this->beta, this->vW1), multiply(1 - this->beta, dW1));
    vector<vector<double>> newsW1 = add(multiply(this->beta2, this->sW1),
                                        multiply(1 - this->beta, multiply(dW1, dW1)));
    this->W1 = subtract(this->W1, multiply(this->learning_rate, divide(newvW1, add(sqrt(newsW1), this->epsilon))));
    this->vW1 = newvW1;
    this->sW1 = newsW1;

    vector<double> newvb1 = add(multiply_bias(this->beta, this->vb1), multiply_bias(1 - this->beta, db1));
    vector<double> newsb1 = add(multiply_bias(this->beta2, this->sb1),
                                multiply_bias(1 - this->beta, multiply_bias(db1, db1)));
    this->b1 = subtract_bias(this->b1,
                             multiply_bias(this->learning_rate, divide(newvb1, add_bias(sqrt(newsb1), this->epsilon))));
    this->vb1 = newvb1;
    this->sb1 = newsb1;

    vector<vector<double>> newvW2 = add(multiply(this->beta, this->vW2), multiply(1 - this->beta, dW2));
    vector<vector<double>> newsW2 = add(multiply(this->beta2, this->sW2),
                                        multiply(1 - this->beta, multiply(dW2, dW2)));
    this->W2 = subtract(this->W2, multiply(this->learning_rate, divide(dW2, add(sqrt(newsW2), this->epsilon))));
    this->vW2 = newvW2;
    this->sW2 = newsW2;

    vector<double> newvb2 = add(multiply_bias(this->beta, this->vb2), multiply_bias(1 - this->beta, db2));
    vector<double> newsb2 = add(multiply_bias(this->beta, this->sb2),
                                multiply_bias(1 - this->beta2, multiply_bias(db2, db2)));
    this->b2 = subtract_bias(this->b2,
                             multiply_bias(this->learning_rate, divide(newvb2, add_bias(sqrt(newsb2), this->epsilon))));
    this->vb2 = newvb2;
    this->sb2 = newsb2;


    vector<vector<double>> newvW3 = add(multiply(this->beta, this->vW3), multiply(1 - this->beta, dW3));
    vector<vector<double>> newsW3 = add(multiply(this->beta2, this->sW3),
                                        multiply(1 - this->beta, multiply(dW3, dW3)));
    this->W3 = subtract(this->W3, multiply(this->learning_rate, divide(dW3, add(sqrt(newsW3), this->epsilon))));
    this->vW3 = newvW3;
    this->sW3 = newsW3;

    vector<double> newvb3 = add(multiply_bias(this->beta, this->vb3), multiply_bias(1 - this->beta, db3));
    vector<double> newsb3 = add(multiply_bias(this->beta2, this->sb3),
                                multiply_bias(1 - this->beta, multiply_bias(db3, db3)));
    this->b3 = subtract_bias(this->b3,
                             multiply_bias(this->learning_rate, divide(newvb3, add_bias(sqrt(newsb3), this->epsilon))));
    this->vb3 = newvb3;
    this->sb3 = newsb3;
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
    for (int i = 0; i < epochs; i++) {
        clock_t start = clock();
        for (size_t j = 0; j < input.size(); j += BATCH_SIZE) {
            vector<vector<double>> mini_batch = {};
            vector<vector<double>> mini_batch_label = {};
            for (size_t k = j; k < j + BATCH_SIZE; k++) {
                mini_batch.emplace_back(input[k]);
                mini_batch_label.emplace_back(targets[k]);
            }
            this->backward_propagation(mini_batch, mini_batch_label);
        }
//        this->backward_propagation(input, targets);
        clock_t end = clock();

        cout << "Last epoch took: " << double(end - start) / CLOCKS_PER_SEC << " s" << endl;
        cout << "Accuracy: " << dnn::accuracy(predict(input), targets) << endl;
        this->setLR(this->getLR() * 0.9);
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

double dnn::loss(const vector<vector<double>> &predicted, const vector<vector<double>> &ground_truth) {
    if (predicted.size() != ground_truth.size()) {
        throw invalid_argument("The predicted input should have the same size as the ground truth!");
    }
    double loss = 0;
    for (size_t i = 0; i < predicted.size(); i++) {
        for (size_t j = 0; j < predicted[i].size(); j++) {
            loss += ground_truth[i][j] * log(predicted[i][j]);
        }
    }
    return -loss / predicted.size();
}

vector<vector<double>> dnn::predict(const vector<vector<double>> &input) {
    ForwardPassOutput forward_pass_output;
    forward_propagation(input, forward_pass_output);
    return forward_pass_output.A3;
}
