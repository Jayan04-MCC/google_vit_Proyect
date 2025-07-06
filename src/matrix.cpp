//
// Created by JAYAN on 05/07/2025.
//

#include "../include/matrix.h"
#include <fstream>
#include <sstream>

// Constructores
Matrix::Matrix() : rows(0), cols(0) {}

Matrix::Matrix(size_t r, size_t c, float init_val)
    : rows(r), cols(c), data(r * c, init_val) {}

// imprimir filas y columnas
void Matrix::getRowsCols() const {
    std::cout << "Filas: " << rows << ", Columnas: " << cols << std::endl;
}
// Indexación
float& Matrix::operator()(size_t i, size_t j) {
    if (i >= rows || j >= cols) {
        std::cerr << "Acceso fuera de rango: (" << i << ", " << j << ") "
                  << "en una matriz de " << rows << " x " << cols << std::endl;
    }
    assert(i < rows && j < cols);
    return data[i * cols + j];
}

float Matrix::operator()(size_t i, size_t j) const {
    assert(i < rows && j < cols);
    return data[i * cols + j];
}

// Impresión básica
void Matrix::print(size_t max_rows, size_t max_cols) const {
    for (size_t i = 0; i < std::min(rows, max_rows); ++i) {
        for (size_t j = 0; j < std::min(cols, max_cols); ++j) {
            std::cout << (*this)(i, j) << "\t";
        }
        std::cout << "\n";
    }
    std::cout << "...\n";
}

// Suma
Matrix Matrix::operator+(const Matrix& other) const {
    // Caso suma directa
    if (rows == other.rows && cols == other.cols) {
        Matrix result(rows, cols);
        for (size_t i = 0; i < data.size(); ++i)
            result.data[i] = data[i] + other.data[i];
        return result;
    }

    // Caso broadcast por fila
    if (other.rows == 1 && other.cols == cols) {
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result(i, j) = (*this)(i, j) + other(0, j);
            }
        }
        return result;
    }

    throw std::runtime_error("Suma de matrices incompatible: dimensiones no compatibles ni broadcastables");
}

// Producto escalar
Matrix Matrix::operator*(float scalar) const {
    Matrix result(rows, cols);
    for (size_t i = 0; i < data.size(); ++i)
        result.data[i] = data[i] * scalar;
    return result;
}

// Multiplicación de matrices
Matrix Matrix::operator*(const Matrix& other) const {

    assert(cols == other.rows);
    Matrix result(rows, other.cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < other.cols; ++j) {
            float sum = 0;
            for (size_t k = 0; k < cols; ++k)
                sum += (*this)(i, k) * other(k, j);
            result(i, j) = sum;
        }
    }
    return result;
}

// Transpuesta
Matrix Matrix::transpose() const {
    Matrix result(cols, rows);
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            result(j, i) = (*this)(i, j);
    return result;
}

// Aplicar función elemento a elemento
Matrix Matrix::apply(float (*func)(float)) const {
    Matrix result(rows, cols);
    for (size_t i = 0; i < data.size(); ++i)
        result.data[i] = func(data[i]);
    return result;
}

// Softmax por filas
Matrix Matrix::softmax_rows() const {
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        float max_val = -INFINITY;
        for (size_t j = 0; j < cols; ++j)
            max_val = std::max(max_val, (*this)(i, j));

        float sum = 0.0f;
        for (size_t j = 0; j < cols; ++j)
            sum += std::exp((*this)(i, j) - max_val);

        for (size_t j = 0; j < cols; ++j)
            result(i, j) = std::exp((*this)(i, j) - max_val) / sum;
    }
    return result;
}

// Normalización por fila (LayerNorm simplificada)
Matrix Matrix::layer_norm(float epsilon) const {
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        float mean = 0.0f, var = 0.0f;
        for (size_t j = 0; j < cols; ++j)
            mean += (*this)(i, j);
        mean /= cols;
        for (size_t j = 0; j < cols; ++j)
            var += ((*this)(i, j) - mean) * ((*this)(i, j) - mean);
        var /= cols;
        float stddev = std::sqrt(var + epsilon);
        for (size_t j = 0; j < cols; ++j)
            result(i, j) = ((*this)(i, j) - mean) / stddev;
    }
    return result;
}

// Cargar CSV (sin encabezados)
Matrix Matrix::fromCSV(const std::string& filename, bool skipHeader) {
    std::ifstream file(filename);
    std::string line;

    std::vector<std::vector<float>> values;

    if (skipHeader && std::getline(file, line)) {
        // Ignora la primera línea (header)
    }

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<float> row;

        while (std::getline(ss, cell, ',')) {
            row.push_back(std::stof(cell));
        }

        values.push_back(row);
    }

    size_t rows = values.size();
    size_t cols = values[0].size();
    Matrix result(rows, cols);

    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            result(i, j) = values[i][j];

    return result;
}

Matrix Matrix::sliceCols(size_t start_col, size_t end_col) const {
    assert(start_col < end_col && end_col <= cols);
    size_t new_cols = end_col - start_col;
    Matrix result(rows, new_cols);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < new_cols; ++j) {
            result(i, j) = (*this)(i, start_col + j);
        }
    }
    return result;
}

void Matrix::setCols(size_t start_col, const Matrix& src) {
    assert(src.rows == rows);
    assert(start_col + src.cols <= cols);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < src.cols; ++j) {
            (*this)(i, start_col + j) = src(i, j);
        }
    }
}



