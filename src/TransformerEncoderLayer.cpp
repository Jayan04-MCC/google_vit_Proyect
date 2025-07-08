//
// Created by JAYAN on 08/07/2025.
//

#include "../include/TransformerEncoderLayer.h"
#include "../include/Gelu.h"

TransformerEncoderLayer::TransformerEncoderLayer(
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
)
    : norm1(norm1_weight_path, norm1_bias_path),
      mha(mha_Wq, mha_Bq, mha_Wk, mha_Bk, mha_Wv, mha_Bv, mha_Wo, mha_Bo, num_heads),
      norm2(norm2_weight_path, norm2_bias_path),
      intermediate(intermediate_weight_path, intermediate_bias_path),
      output_dense(output_dense_weight_path, output_dense_bias_path)
{}

Matrix TransformerEncoderLayer::forward(const Matrix& input) {
    input.getRowsCols();
    Matrix x1 = norm1.forward(input);
    Matrix attn_out = mha.forward(x1);
    Matrix x2 = input + attn_out;  // Residual

    Matrix x3 = norm2.forward(x2);
    Matrix mlp_out = intermediate.forward(x3);
    mlp_out = Gelu::forward(mlp_out);
    mlp_out = output_dense.forward(mlp_out);

    return x2 + mlp_out;  // Segundo residual
}
