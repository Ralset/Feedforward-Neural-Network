#include <nn/NeuralNetwork.h>

#include "nn/ActivationFunctions.h"
#include "nn/Layer.h"

NeuralNetwork::NeuralNetwork(const std::vector<int>& layer_sizes) {
    for (int i = 1; i < static_cast<int>(layer_sizes.size()); i++) {
        layers.push_back(Layer(layer_sizes[i - 1], layer_sizes[i]));
    }
}

void NeuralNetwork::AddLayer(const int layer_size) {
    layers.push_back(Layer(layers.back().getOutputSize(), layer_size));
}

Eigen::VectorXd NeuralNetwork::ForwardPass(const Eigen::VectorXd& input) {
    Eigen::VectorXd activations = input;

    for (int i = 0; i < static_cast<int>(layers.size()) - 1; i++) {
        activations = ActFun::apply(ActivationFunctionType::ReLU,
                                    layers[i].ForwardPass(activations));
    }

    // For output passing trough a sigmoid function
    activations = ActFun::apply(ActivationFunctionType::Sigmoid,
                                layers.back().ForwardPass(activations));

    return activations;
}
