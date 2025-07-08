//
// Created by JAYAN on 05/07/2025.
//

#ifndef PATCHEMBEDDER_H
#define PATCHEMBEDDER_H

#include "matrix.h"
#include <string>

class PatchEmbedder {
private:
    Matrix weight;  // W: 768 x 768
    Matrix bias;    // b: 1 x 768

public:
    PatchEmbedder(const std::string& weightPath, const std::string& biasPath);

    // Aplica el embedding a una matriz de parches (196 x 768)
    Matrix embed(const Matrix& patches);
    
    // Versión CUDA optimizada
    Matrix embed_cuda(const Matrix& patches);
};


#endif //PATCHEMBEDDER_H
