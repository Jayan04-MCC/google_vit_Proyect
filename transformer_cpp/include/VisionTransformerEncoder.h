//
// Created by JAYAN on 08/07/2025.
//

#ifndef VISIONTRANSFORMERENCODER_H
#define VISIONTRANSFORMERENCODER_H

#pragma once

#include <vector>
#include <string>
#include "TransformerEncoderLayer.h"

class VisionTransformerEncoder {
private:
    std::vector<TransformerEncoderLayer> layers;

public:
    VisionTransformerEncoder(const std::string& weight_dir, int num_layers, int num_heads);
    Matrix forward(const Matrix& input);
};

#endif //VISIONTRANSFORMERENCODER_H
