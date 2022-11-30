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
    auto ih = image_loader(image_path, label_path, 1000);
    clock_t end = clock();
    cout << "Time: " << double(end - start) / CLOCKS_PER_SEC << " s" << endl;

    cout << "Loaded " << ih.image_count <<" images, with " << ih.label_count << " labels" << endl;

    cout << "Initializing network" << endl;
    auto nn = dnn();
    cout << "Starting training" << endl;
    auto images = ih.get_all_images();
    auto labels = ih.get_all_labels();
    nn.gradient_descent(images, labels, 100);
    cout << "Training finished" << endl;
    vector<vector<double>> predictions = nn.predict(images);
    cout << "Accuracy: " << dnn::accuracy(predictions, labels) << endl;


    return 0;
}
