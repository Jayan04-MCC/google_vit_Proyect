//
// Created by JAYAN on 06/07/2025.
//

#ifndef INPUTBUILDER_H
#define INPUTBUILDER_H

#endif //INPUTBUILDER_H

#include "Matrix.h"
#include <string>

class InputBuilder {
private:
    Matrix cls_token;
    Matrix position_embeddings;

public:
    InputBuilder(const std::string& cls_path, const std::string& pos_path);

    Matrix build(const Matrix& embedded_patches); // Devuelve input final 197x768
};
