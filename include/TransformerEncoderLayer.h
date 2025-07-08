//
// Created by JAYAN on 08/07/2025.
//

#ifndef TRANSFORMERENCODERLAYER_H
#define TRANSFORMERENCODERLAYER_H

#pragma once

#include "Matrix.h"
#include "LayerNorm.h"
#include "MultiHeadSelfAttention.h"
#include "Dense.h"

class TransformerEncoderLayer {
private:
    LayerNorm norm1;
    MultiHeadSelfAttention mha;
    LayerNorm norm2;
    Dense intermediate;
    Dense output_dense;

public:
    TransformerEncoderLayer(
        const std::string& norm1_weight_path,
        const std::string& norm1_bias_path,
        const std::string& mha_Wq, const std::string& mha_Bq,
        const std::string& mha_Wk, const std::string& mha_Bk,
        const std::string& mha_Wv, const std::string& mha_Bv,
        const std::string& mha_Wo, const std::string& mha_Bo,
        const std::string& norm2_weight_path,
        const std::string& norm2_bias_path,
        const std::string& intermediate_weight_path,
        const std::string& intermediate_bias_path,
        const std::string& output_dense_weight_path,
        const std::string& output_dense_bias_path,
        int num_heads
    );

    Matrix forward(const Matrix& input);
};


#endif //TRANSFORMERENCODERLAYER_H
