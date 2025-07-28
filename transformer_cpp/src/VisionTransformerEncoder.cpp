//
// Created by JAYAN on 08/07/2025.
//

#include "../include/VisionTransformerEncoder.h"

VisionTransformerEncoder::VisionTransformerEncoder(const std::string& weight_dir, int num_layers, int num_heads) {
    for (int i = 0; i < num_layers; ++i) {
        std::string prefix = weight_dir + "/vit_encoder_layer_" + std::to_string(i) + "_";

        TransformerEncoderLayer layer(
            prefix + "layernorm_before_weight.csv",
            prefix + "layernorm_before_bias.csv",

            prefix + "attention_attention_query_weight.csv",
            prefix + "attention_attention_query_bias.csv",
            prefix + "attention_attention_key_weight.csv",
            prefix + "attention_attention_key_bias.csv",
            prefix + "attention_attention_value_weight.csv",
            prefix + "attention_attention_value_bias.csv",
            prefix + "attention_output_dense_weight.csv",
            prefix + "attention_output_dense_bias.csv",

            prefix + "layernorm_after_weight.csv",
            prefix + "layernorm_after_bias.csv",

            prefix + "intermediate_dense_weight.csv",
            prefix + "intermediate_dense_bias.csv",
            prefix + "output_dense_weight.csv",
            prefix + "output_dense_bias.csv",

            num_heads
        );

        layers.push_back(std::move(layer));
    }
}

Matrix VisionTransformerEncoder::forward(const Matrix& input) {
    Matrix out = input;
    for (auto& layer : layers) {
        out = layer.forward(out);
    }
    return out;
}

