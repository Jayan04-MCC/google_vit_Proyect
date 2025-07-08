//
// Created by JAYAN on 08/07/2025.
//

// LayerNorm.cpp
#include "../include/LayerNorm.h"
#include <cmath>

LayerNorm::LayerNorm(const std::string& gamma_path, const std::string& beta_path)
    : gamma(Matrix::fromCSV(gamma_path, true)),
      beta(Matrix::fromCSV(beta_path, true)) {}



Matrix LayerNorm::forward(const Matrix& input) {
    size_t rows = input.rows;
    size_t cols = input.cols;
    Matrix output(rows, cols);

    for (size_t i = 0; i < rows; ++i) {
        float mean = 0.0f;
        float var = 0.0f;

        // Calcular media
        for (size_t j = 0; j < cols; ++j)
            mean += input(i, j);
        mean /= cols;

        // Calcular varianza
        for (size_t j = 0; j < cols; ++j)
            var += (input(i, j) - mean) * (input(i, j) - mean);
        var /= cols;

        float epsilon = 1e-5f;

        if (gamma.rows == 768 && gamma.cols == 1)
            gamma = gamma.transpose();  // Ahora es 1 x 768

        if (beta.rows == 768 && beta.cols == 1)
            beta = beta.transpose();
        // Normalizar + escalar y sesgar
        for (size_t j = 0; j < cols; ++j) {
            float norm = (input(i, j) - mean) / std::sqrt(var + epsilon);

            output(i, j) = norm * gamma(0, j) + beta(0, j);
        }
    }

    return output;
}
