#include <iostream>
#include <string>
#include "headers/image_loader.h"
#include "headers/dnn.h"

using namespace std;

int main() {
    cout << "Loading data" << endl;
    clock_t start = clock();
    string image_path = "../data/fashion_mnist_train_vectors.csv";
    string label_path = "../data/fashion_mnist_train_labels.csv";
    auto ih = image_loader(image_path, label_path);
    clock_t end = clock();
    cout << "Time: " << double(end - start) / CLOCKS_PER_SEC << " s" << endl;

    cout << "Loaded " << ih.image_count <<" images, with " << ih.label_count << " labels" << endl;

    cout << "Initializing network" << endl;
    auto nn = dnn();
    cout << "Starting training" << endl;
    vector<vector<double>> images = ih.get_all_images();
    vector<vector<double>> labels = ih.get_all_labels();
    nn.gradient_descent(images, labels, 5);
    cout << "Training finished" << endl << endl;

    // Accuracy on test set
    string test_image_path = "../data/fashion_mnist_test_vectors.csv";
    string test_label_path = "../data/fashion_mnist_test_labels.csv";
    auto test_images_loader = image_loader(test_image_path, test_label_path);
    vector<vector<double>> test_images = test_images_loader.get_all_images();
    vector<vector<double>> test_labels = test_images_loader.get_all_labels();
    vector<vector<double>> predictions = nn.predict(test_images);
    cout << "Accuracy on test set: " << dnn::accuracy(predictions, test_labels) << endl;

    return 0;
}
