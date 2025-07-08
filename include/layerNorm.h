//
// Created by JAYAN on 08/07/2025.
//

#ifndef LAYERNORM_H
#define LAYERNORM_H

// LayerNorm.h
#pragma once
#include "matrix.h"

class LayerNorm {
private:
    Matrix gamma;  // Peso (weight)
    Matrix beta;   // Sesgo (bias)

public:
    LayerNorm(const std::string& gamma_path, const std::string& beta_path);
    Matrix forward(const Matrix& input);
    Matrix get_gamma() const;
    Matrix get_beta() const;
};


#endif //LAYERNORM_H
