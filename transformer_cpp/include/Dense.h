//
// Created by JAYAN on 08/07/2025.
//

#ifndef DENSE_H
#define DENSE_H

// Dense.h
#pragma once
#include "matrix.h"

class Dense {
private:
    Matrix weight;
    Matrix bias;

public:
    Dense(const std::string& weight_path, const std::string& bias_path);
    Matrix forward(const Matrix& input);
};


#endif //DENSE_H
