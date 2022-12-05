//
// Created by Matej Hakoš on 11/29/2022.
//

#include <thread>
#include "headers/math.h"

vector<vector<double>> transpose(const vector<vector<double>> &input) {
    vector<vector<double>> output;
    for (size_t i = 0; i < input[0].size(); i++) {
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
            for (size_t k = 0; k < i.size(); k++) {
                sum += i[k] * j[k];
            }
            row.push_back(sum);
        }
        output.push_back(row);
    }
    return output;
}

void addHalf(const vector<vector<double>> &input1, const vector<double> &input2,
             vector<vector<double>> &output,
             int begin, int end) {
    for (int i = begin; i < end; i++) {
        vector<double> row;
        for (size_t j = 0; j < input1[0].size(); j++) {
            row.push_back(input1[i][j] + input2[j]);
        }
        output.push_back(row);
    }
}

vector<vector<double>> add(const vector<vector<double>> &input1, const vector<double> &input2) {
    vector<vector<double>> output;
    for (size_t i = 0; i < input1.size(); i++) {
        vector<double> row;
        for (size_t j = 0; j < input1[0].size(); j++) {
            row.push_back(input1[i][j] + input2[j]);
        }
        output.push_back(row);
    }
    return output;
}

vector<vector<double>> add(const vector<vector<double>> &input1, const vector<vector<double>> &input2) {
    vector<vector<double>> output;
    for (size_t i = 0; i < input1.size(); i++) {
        vector<double> row;
        for (size_t j = 0; j < input1[0].size(); j++) {
            row.push_back(input1[i][j] + input2[i][j]);
        }
        output.push_back(row);
    }
    return output;
}

vector<double> add(const vector<double> &input1, const vector<double> &input2) {
    vector<double> output;
    for (size_t i = 0; i < input1.size(); i++) {
        output.push_back(input1[i] + input2[i]);
    }
    return output;
}

vector<vector<double>> subtract(const vector<vector<double>> &input1, const vector<vector<double>> &input2) {
    vector<vector<double>> output;
    for (size_t i = 0; i < input1.size(); i++) {
        vector<double> row;
        for (size_t j = 0; j < input1[0].size(); j++) {
            row.push_back(input1[i][j] - input2[i][j]);
        }
        output.push_back(row);
    }
    return output;
}

vector<vector<double>> subtract(const vector<double> &input1, const vector<vector<double>> &input2) {
    vector<vector<double>> output;
    for (size_t i = 0; i < input2.size(); i++) {
        vector<double> row;
        for (size_t j = 0; j < input2[0].size(); j++) {
            row.push_back(input1[j] - input2[i][j]);
        }
        output.push_back(row);
    }
    return output;
}

vector<vector<double>> subtract(const vector<vector<double>> &input1, const vector<double> &input2) {
    vector<vector<double>> output;
    for (size_t i = 0; i < input1.size(); i++) {
        vector<double> row;
        for (size_t j = 0; j < input1[0].size(); j++) {
            row.push_back(input1[i][j] - input2[j]);
        }
        output.push_back(row);
    }
    return output;
}

vector<double> subtract_bias(const vector<double> &input1, const vector<double> &input2) {
    vector<double> output;
    for (size_t i = 0; i < input1.size(); i++) {
        output.push_back(input1[i] - input2[i]);
    }
    return output;
}

vector<vector<double>> multiply(double input1, const vector<vector<double>> &input2) {
    vector<vector<double>> output;
    for (size_t i = 0; i < input2.size(); i++) {
        vector<double> row;
        for (size_t j = 0; j < input2[0].size(); j++) {
            row.push_back(input1 * input2[i][j]);
        }
        output.push_back(row);
    }
    return output;
}

vector<vector<double>> multiply(const vector<vector<double>> &input1, const vector<vector<double>> &input2) {
    vector<vector<double>> output;
    for (size_t i = 0; i < input1.size(); i++) {
        vector<double> row;
        for (size_t j = 0; j < input1[0].size(); j++) {
            row.push_back(input1[i][j] * input2[i][j]);
        }
        output.push_back(row);
    }
    return output;
}

vector<double> sum(const vector<vector<double>> &input) {
    vector<double> output;
    for (size_t i = 0; i < input[0].size(); i++) {
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

vector<vector<double>> divide(const vector<vector<double>> &input1, const vector<vector<double>> &input2) {
    vector<vector<double>> output;
    for (size_t i = 0; i < input1.size(); i++) {
        vector<double> row;
        for (size_t j = 0; j < input1[0].size(); j++) {
            row.push_back(input1[i][j] / input2[i][j]);
        }
        output.push_back(row);
    }
    return output;
}

vector<double> divide(const vector<double> &input1, const vector<double> &input2) {
    vector<double> output;
    for (size_t i = 0; i < input1.size(); i++) {
        output.push_back(input1[i] / input2[i]);
    }
    return output;
}

vector<vector<double>> sqrt(const vector<vector<double>> &input) {
    vector<vector<double>> output;
    for (size_t i = 0; i < input.size(); i++) {
        vector<double> row;
        for (size_t j = 0; j < input[0].size(); j++) {
            row.push_back(std::sqrt(input[i][j]));
        }
        output.push_back(row);
    }
    return output;
}

vector<double> multiply_bias(const vector<double> &input1, const vector<double> &input2) {
    vector<double> output;
    for (size_t i = 0; i < input1.size(); i++) {
        output.push_back(input1[i] * input2[i]);
    }
    return output;
}

vector<vector<double>> add(const vector<vector<double>> &input1, double input2) {
    vector<vector<double>> output;
    for (size_t i = 0; i < input1.size(); i++) {
        vector<double> row;
        for (size_t j = 0; j < input1[0].size(); j++) {
            row.push_back(input1[i][j] + input2);
        }
        output.push_back(row);
    }
    return output;
}

vector<double> add_bias(const vector<double> &input1, double input2) {
    vector<double> output;
    for (double i : input1) {
        output.push_back(i + input2);
    }
    return output;
}

vector<double> sqrt(const vector<double> &input) {
    vector<double> output;
    for (double i : input) {
        output.push_back(std::sqrt(i));
    }
    return output;
}