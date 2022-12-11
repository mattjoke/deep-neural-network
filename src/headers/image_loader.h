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

    // For hyperparameter checks
    vector<vector<double>> validation_image;
    vector<vector<double>> validation_label;

    vector<double> means;
    vector<double> variances;

    // Indices, by which the Image Holder "picks" a batch
    vector<double> indices;

    // Number of images and labels actually loaded
    size_t load_limit;
public:
    size_t image_count = 0;
    size_t label_count = 0;
    size_t validation_count = 0;

    image_loader(const string &images_path, const string &labels_path, size_t load_limit = -1);

    void load_data(string const &images_path, string const &labels_path);

    void normaliseImages();

    void normaliseImages(const vector<double> &mean, const vector<double> &variance);

    vector<vector<double>> get_all_images();

    vector<vector<double>> get_all_images_unshuffled();

    vector<vector<double>> get_all_labels();

    vector<vector<double>> get_all_labels_unshuffled();

    vector<vector<double>> get_images(int index);

    vector<vector<double>> get_images(int from, int to);

    vector<vector<double>> get_labels(int index);

    vector<vector<double>> get_labels(int from, int to);

    void split_to_validation();

    void shuffle();

    const vector<double> &getMeans() const;

    const vector<double> &getVariances() const;
};


#endif //DEEP_NEURAL_NETWORK_IMAGE_LOADER_H
