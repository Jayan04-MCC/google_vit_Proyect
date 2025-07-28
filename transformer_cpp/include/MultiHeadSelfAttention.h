//
// Created by JAYAN on 06/07/2025.
//

#ifndef MULTIHEADSELFATTENTION_H
#define MULTIHEADSELFATTENTION_H

#include "Matrix.h"
#include <string>
class MultiHeadSelfAttention {
private:
    size_t model_dim;
    size_t num_heads;
    size_t head_dim;

    // Pesos por cabeza (ya cargados como matriz completa, se dividen internamente)
    Matrix Wq, Bq;
    Matrix Wk, Bk;
    Matrix Wv, Bv;
    Matrix Wo, Bo;

    Matrix split_heads(const Matrix& X);
    Matrix scaled_dot_product_attention(const Matrix& Q, const Matrix& K, const Matrix& V);
    Matrix softmax(const Matrix& M);

public:
    MultiHeadSelfAttention(
        const std::string& Wq_path, const std::string& Bq_path,
        const std::string& Wk_path, const std::string& Bk_path,
        const std::string& Wv_path, const std::string& Bv_path,
        const std::string& Wo_path, const std::string& Bo_path,
        size_t num_heads
    );

    Matrix forward(const Matrix& input); // input: (197 x 768)
};

#endif //MULTIHEADSELFATTENTION_H
