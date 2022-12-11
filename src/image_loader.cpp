//
// Created by Matej Hakoš on 11/28/2022.
//

#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>

#include "headers/image_loader.h"

using namespace std;

image_loader::image_loader(const string &images_path, const string &labels_path, size_t load_limit) {
    this->load_limit = load_limit;
    this->load_data(images_path, labels_path);

    // Create an index vector, which will be used to shuffle the images and labels
    this->indices = vector<double>(this->image_count);
    for (size_t i = 0; i < image_count; ++i) {
        this->indices[i] = static_cast<double>(i);
    }

    this->shuffle();
}


void image_loader::load_data(string const &images_path, string const &labels_path) {
    ifstream vector_file(images_path);
    ifstream label_file(labels_path);

    if (vector_file.fail() || label_file.fail()) {
        cerr << "Error opening files, check your inputs." << endl;
        exit(1);
    }

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
        // Store a vector
        image.push_back(stod(value));
        this->image_count++;
        this->images.push_back(image);

        // Create a label vector
        vector<double> one_hot_encoded(10, 0);
        int index = static_cast<int>(stod(label_line));
        one_hot_encoded[index] = 1;
        this->labels.emplace_back(one_hot_encoded);
        this->label_count++;

        // Break if we reached the load limit
        if (this->image_count >= this->load_limit) {
            break;
        }
    }
}

void image_loader::normaliseImages() {
    static constexpr size_t IMGSIZE = 784;
    // Compute sum of values for each pixel
    vector<double> means_m(IMGSIZE, 0);
    for (const auto& image: images) {
        for (size_t i = 0; i < IMGSIZE; i++) {
            means_m[i] += image[i];
        }
    }
    // Divide by number of images to get the mean
    for (double & mean : means_m) {
        mean = mean / images.size();
    }

    // Compute sum of squared differences from mean
    vector<double> variances_m(IMGSIZE, 0);
    for (const auto& image : images) {
        for (size_t i = 0; i < IMGSIZE; i++) {
            means_m[i] += pow(image[i] - means_m[i], 2);
        }
    }
    // Divide by number of images to get variance
    for (double & stddev : variances_m) {
        stddev = stddev / images.size();
    }

    // Normalise data using the mean and variance
    for (auto image : images) {
        for (size_t i = 0; i < IMGSIZE; i++) {
            image[i] = (image[i] - means_m[i]) / variances_m[i];
        }
    }
    means = means_m;
    variances = variances_m;
}

void image_loader::normaliseImages(const vector<double>& mean, const vector<double>& variance) {
    // Normalise data using already computed mean and variance (values computed from the training set)
    for (auto image : images) {
        for (size_t i = 0; i < images[0].size(); i++) {
            image[i] = (image[i] - mean[i]) / variance[i];
        }
    }
}

vector<vector<double>> image_loader::get_all_images() {
    vector<vector<double>> output;
    for (auto &i: this->indices) {
        output.emplace_back(this->images[static_cast<int>(i)]);
    }
    return output;
}

vector<vector<double>> image_loader::get_all_images_unshuffled() {
    return this->images;
}

vector<vector<double>> image_loader::get_all_labels() {
    vector<vector<double>> output;
    for (auto &i: this->indices) {
        output.emplace_back(this->labels[static_cast<int>(i)]);
    }
    return output;
}

vector<vector<double>> image_loader::get_all_labels_unshuffled() {
    return this->labels;
}

vector<vector<double>> image_loader::get_images(const int index){
    return get_images(0, index);
}

vector<vector<double>> image_loader::get_images(const int from, const int to) {
    vector<vector<double>> image;
    for (int i = from; i < to; i++) {
        image.push_back(this->images[i]);
    }
    return image;
}

vector<vector<double>> image_loader::get_labels(const int index) {
    return get_labels(0, index);
}

vector<vector<double>> image_loader::get_labels(const int from, const int to) {
    vector<vector<double>> label;
    for (int i = from; i < to; i++) {
        label.push_back(this->labels[i]);
    }
    return label;
}

void image_loader::split_to_validation() {
    this->validation_count = this->image_count / 20;
    size_t image_cnt = this->image_count;
    size_t diff = this->image_count - this->validation_count;
    for (size_t i = image_cnt - 1; i >= diff; i--) {
        // Store the validation images and labels
        this->validation_image.emplace_back(this->images[i]);
        this->validation_label.emplace_back(this->labels[i]);
        // Remove the validation images and labels from the training set
        this->images.pop_back();
        this->labels.pop_back();
        // Remove the validation indices from the training set
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


const vector<double> &image_loader::getMeans() const {
    return means;
}

const vector<double> &image_loader::getVariances() const {
    return variances;
}





