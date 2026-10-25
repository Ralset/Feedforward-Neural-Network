#include <iostream>
#include <vector>

#include "nn/NeuralNetwork.h"

using namespace std;

int main() {
    NeuralNetwork nn;

    nn.AddLayer(2);
    nn.AddLayer(5);
    nn.AddLayer(5);
    nn.AddLayer(3);
    nn.AddLayer(1);

    Eigen::VectorXd input{{2.5, 30.0}};
    cout << "Result of the pass: " << nn.Predict(input) << endl;
    return 0;
}