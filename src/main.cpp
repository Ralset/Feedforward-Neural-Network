#include <Eigen/Dense>
#include <iostream>
#include <vector>

using Eigen::MatrixXd;

double ReLU(double x) { return std::max(0.0, x); }

double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

double ReLU_derivative(double x) { return x > 0 ? 1.0 : 0.0; }

double sigmoid_derivative(double x) {
    double s = sigmoid(x);
    return s * (1 - s);
}

int main() {
    int L;
    std::cout << "Enter the number of layers: ";
    std::cin >> L;
    std::vector<int> v(L + 1);
    std::cout << "Input layer size: ";
    std::cin >> v[0];
    std::cout << "Enter the number of neurons for each layer: \n";
    for (int i = 1; i <= L; i++) {
        std::cout << "Layer " << i << ": ";
        std::cin >> v[i];
    }
    std::vector<MatrixXd> weights(L + 1);
    std::vector<MatrixXd> biases(L + 1);
    for (int i = 1; i <= L; i++) {
        weights[i] = MatrixXd::Random(v[i], v[i - 1]);
        biases[i] = MatrixXd::Random(v[i], 1);
        std::cout << "Weights for layer " << i + 1 << " -> layer " << i + 2
                  << ": \n";
        std::cout << weights[i] << std::endl;
    }
    MatrixXd input(v[0], 1);
    for (int i = 0; i < v[0]; i++) {
        std::cout << "Input value " << i + 1 << ": ";
        std::cin >> input(i, 0);
    }
    std::cout << "Input: \n" << input << std::endl;
    // forward pass
    std::vector<MatrixXd> activations(L + 1);
    activations[0] = input;
    for (int i = 1; i < L; i++) {
        activations[i] = weights[i] * activations[i - 1] + biases[i];
        for (int j = 0; j < v[i]; j++)
            activations[i](j, 0) = ReLU(activations[i](j, 0));
    }
    activations[L] = weights[L] * activations[L - 1] + biases[L];
    for (int j = 0; j < v[L]; j++)
        activations[L](j, 0) = sigmoid(activations[L](j, 0));
    std::cout << "Output layer is:" << std::endl;
    std::cout << activations[L] << std::endl;

    return 0;
}