#include <iostream>
#include <string>
#include <fstream>
#include "headers/image_loader.h"
#include "headers/dnn.h"

using namespace std;

int main() {
    cout << "Loading data" << endl;
    clock_t start = clock();
    string image_path = "./data/fashion_mnist_train_vectors.csv";
    string label_path = "./data/fashion_mnist_train_labels.csv";
    auto ih = image_loader(image_path, label_path);
    ih.normaliseImages();
    ih.split_to_validation();
    clock_t end = clock();
    cout << "Time required to load and parse images: " << double(end - start) / CLOCKS_PER_SEC << " s" << endl;

    cout << "Loaded:" << endl << ih.image_count << " images, with " << ih.label_count << " labels" << endl
         << ih.validation_count << " validation images" << endl;

    cout << "Initializing network" << endl;
    auto nn = dnn();
    cout << "Starting training" << endl;
    nn.train(&ih, 10); // Gradient descent, 10 epochs, shuffle every epoch
    cout << "Training finished" << endl << endl;

    cout << "Checking if there is a need to train more" << endl;
    double accuracy = dnn::accuracy(ih.get_all_images_unshuffled(), ih.get_all_labels_unshuffled());
    cout << "Accuracy: " << accuracy << endl;
    while (accuracy < 0.9) {
        cout << "Training more" << endl;
        nn.train(&ih, 2);
        cout << "Checking if there is a need to train more" << endl;
        accuracy = dnn::accuracy(ih.get_all_images_unshuffled(), ih.get_all_labels_unshuffled());
    }

    // Accuracy on test set
    string test_image_path = "./data/fashion_mnist_test_vectors.csv";
    string test_label_path = "./data/fashion_mnist_test_labels.csv";

    auto test_images_loader = image_loader(test_image_path, test_label_path);
    test_images_loader.normaliseImages(ih.getMeans(), ih.getVariances());

    vector<vector<double>> test_images = test_images_loader.get_all_images();
    vector<vector<double>> test_labels = test_images_loader.get_all_labels();
    vector<vector<double>> predictions = nn.predict(test_images);
    cout << "Accuracy on test set: " << dnn::accuracy(predictions, test_labels) << endl;

    // Print predictions to `train_predictions.csv` and `test_predictions.csv`
    string output_training_path = "./train_predictions.csv";
    string output_test_path = "./test_predictions.csv";

    cout << "Writing predictions to files" << endl;
    ofstream train_image_file(output_training_path);
    ofstream test_image_file(output_test_path);
    if (train_image_file.fail() || test_image_file.fail()) {
        cout << "Failed to create and open output files. Please check your privileges" << endl;
        return 1;
    }

    vector<vector<double>> train_predictions = nn.predict(ih.get_all_images_unshuffled());
    vector<vector<double>> test_predictions = nn.predict(test_images_loader.get_all_images_unshuffled());
    for (auto &i: train_predictions) {
        for (auto &j: i) {
            train_image_file << j << endl;
        }
    }
    for (auto &i: test_predictions) {
        for (auto &j: i) {
            test_image_file << j << endl;
        }
    }

    cout << "All information saved to output files" << endl;
    cout << "Goodbye!" << endl;
    return 0;
}
