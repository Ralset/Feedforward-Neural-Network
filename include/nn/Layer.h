#pragma once

#include <Eigen/Dense>

#include "Eigen/Core"

class Layer {
   public:
    Layer(int input_size, int output_size);
    ~Layer() = default;
    int getInputSize() { return input_size; }
    int getOutputSize() { return output_size; }
    Eigen::VectorXd ForwardPass(const Eigen::VectorXd& activations);

   private:
    int input_size;
    int output_size;
    Eigen::MatrixXd weights;
    Eigen::VectorXd biases;
};