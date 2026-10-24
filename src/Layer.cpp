#include <nn/Layer.h>

#include <iostream>

Layer::Layer(int input_size, int output_size)
    : input_size(input_size), output_size(output_size) {
    weights = Eigen::MatrixXd::Random(output_size, input_size);
    biases = Eigen::VectorXd::Random(output_size);
    std::cout << "Created a layer\n";
    std::cout << "Weights : \n" << weights << '\n';
    std::cout << "Biases : \n" << biases << '\n';
    std::cout << "---\n";
}

Eigen::VectorXd Layer::ForwardPass(const Eigen::VectorXd& input) {
    return weights * input + biases;
}
