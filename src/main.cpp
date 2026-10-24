#include <iostream>
#include <vector>

#include "nn/NeuralNetwork.h"

using namespace std;

int main() {
    int L;
    cout << "Enter the number of layers: ";
    cin >> L;
    vector<int> v(L + 1);
    cout << "Input layer size: ";
    cin >> v[0];
    cout << "Enter the number of neurons for each layer: \n";
    for (int i = 1; i < L; i++) {
        cout << "Layer " << i << ": ";
        cin >> v[i];
    }
    cout << "Enter the number of neurons for output layer: ";
    cin >> v[L];

    NeuralNetwork nn(v);

    return 0;
}