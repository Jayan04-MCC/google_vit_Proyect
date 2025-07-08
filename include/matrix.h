//
// Created by JAYAN on 05/07/2025.
//

#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <iostream>
#include <cassert>
#include <cmath>

class Matrix {
public:
    size_t rows, cols;
    std::vector<float> data;

    Matrix();  // Constructor por defecto
    Matrix(size_t r, size_t c, float init_val = 0.0f);
    void getRowsCols() const;
    float& operator()(size_t i, size_t j);
    float operator()(size_t i, size_t j) const;

    void print(size_t max_rows = 5, size_t max_cols = 5) const;

    // Operaciones
    Matrix operator+(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;
    Matrix operator*(float scalar) const;
    [[nodiscard]] Matrix transpose() const;

    // Funciones útiles
    Matrix apply(float (*func)(float)) const;
    [[nodiscard]] Matrix softmax_rows() const;
    [[nodiscard]] Matrix layer_norm(float epsilon = 1e-5) const;

    static Matrix fromCSV(const std::string& filename, bool skipHeader = false);

    [[nodiscard]] Matrix sliceCols(size_t start_col, size_t end_col) const;
    void setCols(size_t start_col, const Matrix& src);
    [[nodiscard]] Matrix sliceRows(size_t start_row, size_t end_row) const;

    // Funciones CUDA
    Matrix cuda_add(const Matrix& other) const;
    Matrix cuda_scalar_multiply(float scalar) const;
    Matrix cuda_multiply(const Matrix& other) const;
    Matrix cublas_multiply(const Matrix& other) const;

   
};


#endif //MATRIX_H
