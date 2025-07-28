//
// Created by JAYAN on 08/07/2025.
//

// Gelu.cpp
#include "../include/Gelu.h"
#include <cmath>

Matrix Gelu::forward(const Matrix& input) {
    Matrix output(input.rows, input.cols);

    for (size_t i = 0; i < input.rows; ++i) {
        for (size_t j = 0; j < input.cols; ++j) {
            float x = input(i, j);
            long double M_PI = 3.14159265358979323846;
            // Fórmula de GELU aproximada: x * 0.5 * (1 + tanh(√(2/π) * (x + 0.044715x³)))
            float gelu = 0.5f * x * (1.0f + std::tanh(std::sqrt(2.0f / M_PI) * (x + 0.044715f * std::pow(x, 3))));
            output(i, j) = gelu;
        }
    }

    return output;
}
