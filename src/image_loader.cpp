//
// Created by Matej Hakoš on 11/28/2022.
//

#include <fstream>
#include <sstream>
#include <random>

#include "headers/image_loader.h"

using namespace std;

image_loader::image_loader(const string &images_path, const string &labels_path, size_t load_limit) {
    this->load_limit = load_limit;
    this->load_data(images_path, labels_path);

    this->indices = vector<double>(this->image_count);
    for (int i = 0; i < image_count; ++i) {
        this->indices[i] = i;
    }
    //this->split_to_validation();
    this->shuffle();
}


void image_loader::load_data(string const &images_path, string const &labels_path) {
    ifstream vector_file(images_path);
    ifstream label_file(labels_path);

    string vector_line;
    string label_line;
    while (getline(vector_file, vector_line) && getline(label_file, label_line)) {
        vector<double> image;
        string value;
        for (char i: vector_line) {
            if (i != ',') {
                value += i;
            } else {
                image.push_back(stod(value) / 255.0);
                value = "";
            }
        }
        image.push_back(stod(value));
        this->image_count++;
        this->images.push_back(image);

        vector<double> one_hot_encoded(10, 0);
        int index = stod(label_line);
        one_hot_encoded[index] = 1;
        this->labels.emplace_back(one_hot_encoded);
        this->label_count++;

        // Break if we reached the load limit
        if (this->image_count >= this->load_limit) {
            break;
        }
    }
}

vector<vector<double>> image_loader::get_all_images() {
    vector<vector<double>> output;
    for (auto &i: this->indices) {
        output.emplace_back(this->images[i]);
    }
    return output;
}

vector<vector<double>> image_loader::get_all_labels() {
    vector<vector<double>> output;
    for (auto &i: this->indices) {
        output.emplace_back(this->labels[i]);
    }
    return output;
}

vector<vector<double>> image_loader::get_images(int index) {
    return get_images(0, index);
}

vector<vector<double>> image_loader::get_images(int from, int to) {
    vector<vector<double>> image;
    for (int i = from; i < to; i++) {
        image.push_back(this->images[i]);
    }
    return image;
}

vector<vector<double>> image_loader::get_labels(int index) {
    return get_labels(0, index);
}

vector<vector<double>> image_loader::get_labels(int from, int to) {
    vector<vector<double>> label;
    for (int i = from; i < to; i++) {
        label.push_back(this->labels[i]);
    }
    return label;
}

void image_loader::split_to_validation() {
    this->validation_count = this->image_count / 10;
    int image_cnt = this->image_count;
    int diff = this->image_count - this->validation_count;
    for (int i = image_cnt - 1; i >= diff; i--) {
        this->validation_image.emplace_back(this->images[i]);
        this->validation_label.emplace_back(this->labels[i]);
        this->images.pop_back();
        this->labels.pop_back();
        this->image_count--;
        this->label_count--;
    }
    cout << this->image_count << endl;
}

void image_loader::shuffle() {
    random_device rd;
    mt19937 gen(rd());

    std::shuffle(this->indices.begin(), this->indices.end(), gen);
}





