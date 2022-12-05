//
// Created by Matej Hakoš on 11/28/2022.
//

#ifndef DEEP_NEURAL_NETWORK_IMAGE_LOADER_H
#define DEEP_NEURAL_NETWORK_IMAGE_LOADER_H

#include <iostream>
#include <vector>
#include <string>

using namespace std;

class image_loader {
private:
    vector<vector<double>> images;
    vector<vector<double>> labels;

    // FOr hyperparameter checks
    vector<vector<double>> validation_image;
    vector<vector<double>> validation_label;

    vector<double> means;
public:
    const vector<double> &getMeans() const;

    const vector<double> &getVariances() const;

private:
    vector<double> variances;

    // Indices, by which the Image Holder "picks" a batch
    vector<double> indices;

    size_t load_limit;
public:
    size_t image_count{};
    size_t label_count{};

    size_t validation_count{};

    image_loader(const string &images_path, const string &labels_path, size_t load_limit = -1);

    void load_data(string const &images_path, string const &labels_path);

    void normaliseImages();

    void normaliseImages(const vector<double>& mean, const vector<double>& variance);

    vector<vector<double>> get_all_images();

    vector<vector<double>> get_all_labels();

    vector<vector<double>> get_images(int index);

    vector<vector<double>> get_images(int from, int to);

    vector<vector<double>> get_labels(int index);

    vector<vector<double>> get_labels(int from, int to);

    void split_to_validation();

    void shuffle();
};


#endif //DEEP_NEURAL_NETWORK_IMAGE_LOADER_H
