//
// Created by Matej Hakoš on 11/29/2022.
//

#include "headers/math.h"

vector<vector<double>> transpose(const vector<vector<double>> &input) {
    vector<vector<double>> output;
    for (int i = 0; i < input[0].size(); i++) {
        vector<double> row;
        for (const auto &j: input) {
            row.push_back(j[i]);
        }
        output.push_back(row);
    }
    return output;
}

vector<vector<double>> matmul(const vector<vector<double>> &input1, const vector<vector<double>> &input2) {
    vector<vector<double>> output;
    for (const auto &i: input2) {
        vector<double> row;
        for (const auto &j: transpose(input1)) {
            double sum = 0;
            for (int k = 0; k < i.size(); k++) {
                sum += i[k] * j[k];
            }
            row.push_back(sum);
        }
        output.push_back(row);
    }
    return output;
}

vector<vector<double>> add(const vector<vector<double>> &input1, const vector<double> &input2) {
    vector<vector<double>> output;
    for (int i = 0; i < input1.size(); i++) {
        vector<double> row;
        for (int j = 0; j < input1[0].size(); j++) {
            row.push_back(input1[i][j] + input2[j]);
        }
        output.push_back(row);
    }
    return output;
}

vector<vector<double>> subtract(const vector<vector<double>> &input1, const vector<vector<double>> &input2) {
    vector<vector<double>> output;
    for (int i = 0; i < input1.size(); i++) {
        vector<double> row;
        for (int j = 0; j < input1[0].size(); j++) {
            row.push_back(input1[i][j] - input2[i][j]);
        }
        output.push_back(row);
    }
    return output;
}

vector<vector<double>> subtract(const vector<double> &input1, const vector<vector<double>> &input2) {
    vector<vector<double>> output;
    for (int i = 0; i < input2.size(); i++) {
        vector<double> row;
        for (int j = 0; j < input2[0].size(); j++) {
            row.push_back(input1[j] - input2[i][j]);
        }
        output.push_back(row);
    }
    return output;
}

vector<vector<double>> subtract(const vector<vector<double>> &input1, const vector<double> &input2) {
    vector<vector<double>> output;
    for (int i = 0; i < input1.size(); i++) {
        vector<double> row;
        for (int j = 0; j < input1[0].size(); j++) {
            row.push_back(input1[i][j] - input2[j]);
        }
        output.push_back(row);
    }
    return output;
}

vector<double> subtract_bias(const vector<double> &input1, const vector<double> &input2) {
    vector<double> output;
    for (int i = 0; i < input1.size(); i++) {
        output.push_back(input1[i] - input2[i]);
    }
    return output;
}

vector<vector<double>> multiply(double input1, const vector<vector<double>> &input2) {
    vector<vector<double>> output;
    for (int i = 0; i < input2.size(); i++) {
        vector<double> row;
        for (int j = 0; j < input2[0].size(); j++) {
            row.push_back(input1 * input2[i][j]);
        }
        output.push_back(row);
    }
    return output;
}

vector<vector<double>> multiply(const vector<vector<double>> &input1, const vector<vector<double>> &input2) {
    vector<vector<double>> output;
    for (int i = 0; i < input1.size(); i++) {
        vector<double> row;
        for (int j = 0; j < input1[0].size(); j++) {
            row.push_back(input1[i][j] * input2[i][j]);
        }
        output.push_back(row);
    }
    return output;
}

vector<double> sum(const vector<vector<double>> &input) {
    vector<double> output;
    for (int i = 0; i < input[0].size(); i++) {
        double sum = 0;
        for (const auto &j: input) {
            sum += j[i];
        }
        output.push_back(sum);
    }
    return output;
}

vector<double> multiply_bias(double input1, const vector<double> &input2) {
    vector<double> output;
    for (double i: input2) {
        output.push_back(input1 * i);
    }
    return output;
}
