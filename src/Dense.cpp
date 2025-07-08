//
// Created by JAYAN on 08/07/2025.
//

// Dense.cpp
#include "../include/Dense.h"

Dense::Dense(const std::string& weight_path, const std::string& bias_path)
    : weight(Matrix::fromCSV(weight_path, true)),
      bias(Matrix::fromCSV(bias_path, true).transpose()) {}

Matrix Dense::forward(const Matrix& input) {
    Matrix Wt = weight.transpose();
    return input * Wt + bias;
}
