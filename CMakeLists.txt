cmake_minimum_required(VERSION 3.23)
project(deep_neural_network)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Ofast")

add_executable(deep_neural_network src/main.cpp src/image_loader.cpp src/headers/image_loader.h src/dnn.cpp src/headers/dnn.h src/math.cpp src/headers/math.h)
