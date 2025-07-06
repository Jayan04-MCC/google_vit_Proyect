//
// Created by JAYAN on 05/07/2025.
//

#include "../include/PatchEmbedder.h"

PatchEmbedder::PatchEmbedder(const std::string& weightPath, const std::string& biasPath)
    : weight(Matrix::fromCSV(weightPath,true)), bias(Matrix::fromCSV(biasPath,true)) {}

Matrix PatchEmbedder::embed(const Matrix& patches) {
    Matrix output = patches * weight;  // (196, 768)

    // Broadcast: suma el bias a cada fila
    for (size_t i = 0; i < output.rows; ++i) {
        for (size_t j = 0; j < output.cols; ++j) {
            output(i, j) += bias(j, 0);
        }
    }
    return output;
}
