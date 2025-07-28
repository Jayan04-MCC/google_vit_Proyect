cmake_minimum_required(VERSION 3.30)
project(google_vit)

set(CMAKE_CXX_STANDARD 20)

add_executable(google_vit main.cpp
        include/matrix.h
        src/matrix.cpp
        include/PatchEmbedder.h
        src/PatchEmbedder.cpp
        include/InputBuilder.h
        src/InputBuilder.cpp
        include/MultiHeadSelfAttention.h
        src/MultiHeadSelfAttention.cpp
        src/VisionTransformerEncoder.cpp
        include/VisionTransformerEncoder.h
        include/TransformerEncoderLayer.h
        src/TransformerEncoderLayer.cpp
        include/layerNorm.h
        src/layerNorm.cpp
        include/Dense.h
        src/Dense.cpp
        include/Gelu.h
        src/Gelu.cpp)
