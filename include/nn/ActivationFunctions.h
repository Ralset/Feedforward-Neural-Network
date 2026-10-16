#pragma once

#include <cmath>

struct ActFun {
    static double ReLU(double x) { return std::max(0.0, x); }
    static double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
    static double ReLU_derivative(double x) { return x > 0 ? 1.0 : 0.0; }
    static double sigmoid_derivative(double x) {
        double s = sigmoid(x);
        return s * (1 - s);
    }
};