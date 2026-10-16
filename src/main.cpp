#include <Eigen/Dense>
#include <iostream>
#include <vector>

#include "nn/ActivationFunctions.h"
// #include "nn/Layer.h"
// #include "nn/NeuralNetwork.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

int main() {
    int L;
    cout << "Enter the number of layers: ";
    cin >> L;
    vector<int> v(L + 1);
    cout << "Input layer size: ";
    cin >> v[0];
    cout << "Enter the number of neurons for each layer: \n";
    for (int i = 1; i <= L; i++) {
        cout << "Layer " << i << ": ";
        cin >> v[i];
    }
    vector<MatrixXd> weights(L + 1);
    vector<VectorXd> biases(L + 1);
    for (int i = 1; i <= L; i++) {
        weights[i] = MatrixXd::Random(v[i], v[i - 1]);
        biases[i] = VectorXd::Random(v[i]);
        cout << "Weights for layer " << i + 1 << " -> layer " << i + 2
             << ": \n";
        cout << weights[i] << endl;
    }
    MatrixXd input(v[0], 1);
    for (int i = 0; i < v[0]; i++) {
        cout << "Input value " << i + 1 << ": ";
        cin >> input(i, 0);
    }
    cout << "Input: \n" << input << endl;
    // forward pass
    vector<VectorXd> activations(L + 1);
    activations[0] = input;
    for (int i = 1; i < L; i++) {
        activations[i] = weights[i] * activations[i - 1] + biases[i];
        ActFun::apply(ActivationFunctionType::ReLU, activations[i]);
    }
    activations[L] = weights[L] * activations[L - 1] + biases[L];
    ActFun::apply(ActivationFunctionType::Sigmoid, activations[L]);
    cout << "Output layer is:" << endl;
    cout << activations[L] << endl;

    return 0;
}