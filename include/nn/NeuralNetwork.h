#pragma once

#include <Eigen/Dense>
#include <vector>

#include "nn/Layer.h"

class NeuralNetwork {
   public:
    NeuralNetwork();
    NeuralNetwork(const std::vector<int>& layer_sizes);
    ~NeuralNetwork() = default;
    void AddLayer(const int layer_size);
    Eigen::VectorXd Predict(const Eigen::VectorXd& input);

   private:
    int input_size = -1;
    std::vector<Layer> layers;
    Eigen::VectorXd ForwardPass(const Eigen::VectorXd& input);
};