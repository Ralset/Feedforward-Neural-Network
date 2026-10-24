#pragma once

#include <Eigen/Dense>
#include <vector>

#include "nn/Layer.h"

class NeuralNetwork {
   public:
    NeuralNetwork(const std::vector<int>& layer_sizes);
    ~NeuralNetwork() = default;
    void AddLayer(const int layer_size);
    // for testing its a global function
    Eigen::VectorXd ForwardPass(const Eigen::VectorXd& input);

   private:
    std::vector<Layer> layers;
};