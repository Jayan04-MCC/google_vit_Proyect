//
// Created by JAYAN on 06/07/2025.
//

#include "../include/MultiHeadSelfAttention.h"
#include <cmath>
#include <cassert>
#include <iostream>

MultiHeadSelfAttention::MultiHeadSelfAttention(
    const std::string& Wq_path, const std::string& Bq_path,
    const std::string& Wk_path, const std::string& Bk_path,
    const std::string& Wv_path, const std::string& Bv_path,
    const std::string& Wo_path, const std::string& Bo_path,
    size_t num_heads)
    : Wq(Matrix::fromCSV(Wq_path, true)),
      Bq(Matrix::fromCSV(Bq_path, true)),
      Wk(Matrix::fromCSV(Wk_path, true)),
      Bk(Matrix::fromCSV(Bk_path, true)),
      Wv(Matrix::fromCSV(Wv_path, true)),
      Bv(Matrix::fromCSV(Bv_path, true)),
      Wo(Matrix::fromCSV(Wo_path, true)),
      Bo(Matrix::fromCSV(Bo_path, true)),
      num_heads(num_heads)
{
    model_dim = Wq.cols;
    if (num_heads == 0 || model_dim % num_heads != 0)
        throw std::runtime_error("Error: model_dim no divisible por num_heads");

    head_dim = model_dim / num_heads;
}

Matrix MultiHeadSelfAttention::softmax(const Matrix& M) {
    Matrix result = M;
    for (size_t i = 0; i < M.rows; ++i) {
        float max_val = -1e9;
        for (size_t j = 0; j < M.cols; ++j)
            max_val = std::max(max_val, M(i, j));

        float sum = 0.0f;
        for (size_t j = 0; j < M.cols; ++j) {
            result(i, j) = std::exp(M(i, j) - max_val);
            sum += result(i, j);
        }

        for (size_t j = 0; j < M.cols; ++j)
            result(i, j) /= sum;
    }
    return result;
}

Matrix MultiHeadSelfAttention::forward(const Matrix& input) {
    // input: (197 x 768)
    size_t seq_len = input.rows;
    input.getRowsCols();
    Wq.getRowsCols();
    Bq.getRowsCols();
    Wk.getRowsCols();
    Bk.getRowsCols();
    Wv.getRowsCols();
    Bv.getRowsCols();
    Bq = Bq.transpose();
    Bk = Bk.transpose();
    Bv = Bv.transpose();
    Bq.getRowsCols();
    Bk.getRowsCols();
    Bv.getRowsCols();
    // 1. Linear projection: Q = input * Wq + Bq
    Matrix Q = input * Wq + Bq;
    Matrix K = input * Wk + Bk;
    Matrix V = input * Wv + Bv;

    // 2. Split heads: reshape (197 x 768) → (12, 197, 64)
    std::vector<Matrix> Q_heads, K_heads, V_heads;
    for (size_t h = 0; h < num_heads; ++h) {
        Q_heads.push_back(Q.sliceCols(h * head_dim, (h + 1) * head_dim));
        K_heads.push_back(K.sliceCols(h * head_dim, (h + 1) * head_dim));
        V_heads.push_back(V.sliceCols(h * head_dim, (h + 1) * head_dim));
    }

    std::vector<Matrix> head_outputs;
    float scale = std::sqrt(static_cast<float>(head_dim));

    for (size_t h = 0; h < num_heads; ++h) {
        Matrix Qh = Q_heads[h];  // (197 x 64)
        Matrix Kh = K_heads[h].transpose();  // (64 x 197)
        Matrix scores = (Qh * Kh) * (1.0f / scale);  // (197 x 197)
        Matrix probs = softmax(scores);  // (197 x 197)
        Matrix context = probs * V_heads[h];  // (197 x 64)
        head_outputs.push_back(context);
    }

    // 3. Concatenate heads: (197 x 768)
    Matrix concat(seq_len, model_dim);
    for (size_t h = 0; h < num_heads; ++h) {
        concat.setCols(h * head_dim, head_outputs[h]);
    }

    // 4. Final dense layer
    Bo = Bo.transpose();
    Matrix output = concat * Wo + Bo;  // (197 x 768)
    return output;
}

// Versión CUDA optimizada del forward
Matrix MultiHeadSelfAttention::forward_cuda(const Matrix& input) {
    size_t seq_len = input.rows;
    
    // Preparar bias (transponer una sola vez)
    Matrix Bq_t = Bq.transpose();
    Matrix Bk_t = Bk.transpose();
    Matrix Bv_t = Bv.transpose();
    Matrix Bo_t = Bo.transpose();
    
    // 1. Linear projections con cuBLAS (más eficiente)
    Matrix Q = input.cublas_multiply(Wq);
    Matrix K = input.cublas_multiply(Wk);
    Matrix V = input.cublas_multiply(Wv);
    
    // Agregar bias manualmente
    for (size_t i = 0; i < Q.rows; ++i) {
        for (size_t j = 0; j < Q.cols; ++j) {
            Q(i, j) += Bq_t(j, 0);
            K(i, j) += Bk_t(j, 0);
            V(i, j) += Bv_t(j, 0);
        }
    }
    
    // 2. Procesar todas las cabezas en paralelo
    std::vector<Matrix> head_outputs;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    
    for (size_t h = 0; h < num_heads; ++h) {
        Matrix Qh = Q.sliceCols(h * head_dim, (h + 1) * head_dim);  // (197 x 64)
        Matrix Kh = K.sliceCols(h * head_dim, (h + 1) * head_dim);  // (197 x 64)
        Matrix Vh = V.sliceCols(h * head_dim, (h + 1) * head_dim);  // (197 x 64)
        
        // Usar cuBLAS para multiplicación K^T
        Matrix Kh_t = Kh.transpose();  // (64 x 197)
        Matrix scores = Qh.cublas_multiply(Kh_t).cuda_scalar_multiply(scale);  // (197 x 197)
        
        // Softmax usando la función existente (podría optimizarse más)
        Matrix probs = softmax(scores);  // (197 x 197)
        
        // Contexto final con cuBLAS
        Matrix context = probs.cublas_multiply(Vh);  // (197 x 64)
        head_outputs.push_back(context);
    }
    
    // 3. Concatenar cabezas
    Matrix concat(seq_len, model_dim);
    for (size_t h = 0; h < num_heads; ++h) {
        concat.setCols(h * head_dim, head_outputs[h]);
    }
    
    // 4. Proyección final con cuBLAS
    Matrix output = concat.cublas_multiply(Wo);
    
    // Agregar bias final
    for (size_t i = 0; i < output.rows; ++i) {
        for (size_t j = 0; j < output.cols; ++j) {
            output(i, j) += Bo_t(j, 0);
        }
    }
    
    return output;
}

