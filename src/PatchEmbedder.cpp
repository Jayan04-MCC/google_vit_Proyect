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

// Versión CUDA optimizada
Matrix PatchEmbedder::embed_cuda(const Matrix& patches) {
    // Usar cuBLAS para la multiplicación de matrices (más eficiente para matrices grandes)
    Matrix output = patches.cublas_multiply(weight);  // (196, 768)
    
    // Agregar bias manualmente (más eficiente que crear matriz nueva)
    for (size_t i = 0; i < output.rows; ++i) {
        for (size_t j = 0; j < output.cols; ++j) {
            output(i, j) += bias(j, 0);
        }
    }
    
    return output;
}
