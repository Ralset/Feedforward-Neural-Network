#pragma once

#include <Eigen/Dense>
#include <cmath>

enum class ActivationFunctionType { ReLU, Sigmoid };

struct ActFun {
    static void apply(ActivationFunctionType type, Eigen::VectorXd& x) {
        switch (type) {
            case ActivationFunctionType::ReLU:
                x = x.cwiseMax(0.0);
                break;
            case ActivationFunctionType::Sigmoid:
                x = x.unaryExpr(
                    [](double a) { return 1.0 / (1.0 + std::exp(-a)); });
                break;
        }
    }
};