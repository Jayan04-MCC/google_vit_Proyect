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
        src/MultiHeadSelfAttention.cpp)
