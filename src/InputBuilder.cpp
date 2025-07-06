//
// Created by JAYAN on 06/07/2025.
//

#include "../include/InputBuilder.h"
#include <cassert>

InputBuilder::InputBuilder(const std::string& cls_path, const std::string& pos_path)
    : cls_token(Matrix::fromCSV(cls_path, true)),
      position_embeddings(Matrix::fromCSV(pos_path, true)) {}

Matrix InputBuilder::build(const Matrix& embedded_patches) {
    assert(embedded_patches.cols == cls_token.cols);
    assert(position_embeddings.rows == embedded_patches.rows + 1);
    assert(position_embeddings.cols == cls_token.cols);

    size_t new_rows = embedded_patches.rows + 1;
    size_t cols = embedded_patches.cols;

    Matrix full_input(new_rows, cols);

    // Insertar el CLS token en la primera fila
    for (size_t j = 0; j < cols; ++j)
        full_input(0, j) = cls_token(0, j);

    // Copiar los 196 embeddings
    for (size_t i = 0; i < embedded_patches.rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            full_input(i + 1, j) = embedded_patches(i, j);

    // Sumar position embeddings
    for (size_t i = 0; i < new_rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            full_input(i, j) += position_embeddings(i, j);

    return full_input;
}
