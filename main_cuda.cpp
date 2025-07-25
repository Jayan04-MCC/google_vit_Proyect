#include <iostream>
#include <chrono>
#include "include/matrix.h"
#include "include/PatchEmbedder.h"
#include "include/InputBuilder.h"
#include "include/MultiHeadSelfAttention.h"

int main() {
    std::cout << "=== Google Vision Transformer con CUDA ===" << std::endl;
    
    // Cargar patches
    std::cout << "Cargando patches..." << std::endl;
    Matrix patches = Matrix::fromCSV("C:/Users/JAYAN/CLionProjects/google-vit/patches_00001.csv", false); // (196 x 768)
    
    // Crear PatchEmbedder
    std::cout << "Inicializando PatchEmbedder..." << std::endl;
    PatchEmbedder embedder(
        "C:/Users/JAYAN/CLionProjects/google-vit/pesos/vit_embeddings_patch_embeddings_projection_weight.csv",
        "C:/Users/JAYAN/CLionProjects/google-vit/pesos/vit_embeddings_patch_embeddings_projection_bias.csv"
    );
    
    // Comparar CPU vs CUDA para embedding
    std::cout << "\n=== Patch Embedding ===" << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    Matrix embedded_cpu = embedder.embed(patches);  // (196 x 768)
    auto end = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    start = std::chrono::high_resolution_clock::now();
    Matrix embedded_cuda = embedder.embed_cuda(patches);  // (196 x 768)
    end = std::chrono::high_resolution_clock::now();
    auto cuda_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "CPU Embedding: " << cpu_time.count() << " ms" << std::endl;
    std::cout << "CUDA Embedding: " << cuda_time.count() << " ms" << std::endl;
    std::cout << "Speedup: " << (float)cpu_time.count() / cuda_time.count() << "x" << std::endl;
    
    // Verificar que los resultados son similares
    bool embedding_correct = true;
    for (size_t i = 0; i < std::min(5UL, embedded_cpu.rows) && embedding_correct; ++i) {
        for (size_t j = 0; j < std::min(5UL, embedded_cpu.cols) && embedding_correct; ++j) {
            if (abs(embedded_cpu(i, j) - embedded_cuda(i, j)) > 1e-3) {
                embedding_correct = false;
            }
        }
    }
    std::cout << "Resultados correctos: " << (embedding_correct ? "Si" : "No") << std::endl;
    
    // Usar el resultado CUDA para continuar
    embedded_cuda.print(1, 5);
    std::cout << "Dimensiones embedded: ";
    embedded_cuda.getRowsCols();
    
    // Crear InputBuilder
    std::cout << "\n=== Input Builder ===" << std::endl;
    InputBuilder builder(
        "C:/Users/JAYAN/CLionProjects/google-vit/pesos/vit_embeddings_cls_token.csv",
        "C:/Users/JAYAN/CLionProjects/google-vit/pesos/vit_embeddings_position_embeddingsV2.csv"
    );
    Matrix input = builder.build(embedded_cuda);  // 197 x 768
    
    input.print(1, 5);
    std::cout << "Dimensiones input: ";
    input.getRowsCols();
    
    // Crear MultiHeadSelfAttention
    std::cout << "\n=== Multi-Head Self-Attention ===" << std::endl;
    MultiHeadSelfAttention attention(
        "C:/Users/JAYAN/CLionProjects/google-vit/pesos/vit_encoder_layer_0_attention_attention_query_weight.csv",
        "C:/Users/JAYAN/CLionProjects/google-vit/pesos/vit_encoder_layer_0_attention_attention_query_bias.csv",
        "C:/Users/JAYAN/CLionProjects/google-vit/pesos/vit_encoder_layer_0_attention_attention_key_weight.csv",
        "C:/Users/JAYAN/CLionProjects/google-vit/pesos/vit_encoder_layer_0_attention_attention_key_bias.csv",
        "C:/Users/JAYAN/CLionProjects/google-vit/pesos/vit_encoder_layer_0_attention_attention_value_weight.csv",
        "C:/Users/JAYAN/CLionProjects/google-vit/pesos/vit_encoder_layer_0_attention_attention_value_bias.csv",
        "C:/Users/JAYAN/CLionProjects/google-vit/pesos/vit_encoder_layer_0_attention_output_dense_weight.csv",
        "C:/Users/JAYAN/CLionProjects/google-vit/pesos/vit_encoder_layer_0_attention_output_dense_bias.csv",
        12  // número de cabezas en ViT Base
    );
    
    // Comparar CPU vs CUDA para attention
    start = std::chrono::high_resolution_clock::now();
    Matrix attended_cpu = attention.forward(input);  // (197 x 768)
    end = std::chrono::high_resolution_clock::now();
    cpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    start = std::chrono::high_resolution_clock::now();
    Matrix attended_cuda = attention.forward_cuda(input);  // (197 x 768)
    end = std::chrono::high_resolution_clock::now();
    cuda_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "CPU Attention: " << cpu_time.count() << " ms" << std::endl;
    std::cout << "CUDA Attention: " << cuda_time.count() << " ms" << std::endl;
    std::cout << "Speedup: " << (float)cpu_time.count() / cuda_time.count() << "x" << std::endl;
    
    // Verificar que los resultados son similares
    bool attention_correct = true;
    for (size_t i = 0; i < std::min(5UL, attended_cpu.rows) && attention_correct; ++i) {
        for (size_t j = 0; j < std::min(5UL, attended_cpu.cols) && attention_correct; ++j) {
            if (abs(attended_cpu(i, j) - attended_cuda(i, j)) > 1e-2) {
                attention_correct = false;
            }
        }
    }
    std::cout << "Resultados correctos: " << (attention_correct ? "Si" : "No") << std::endl;
    
    // Mostrar resultado final
    std::cout << "\n=== Resultado Final ===" << std::endl;
    attended_cuda.print(1, 5);
    std::cout << "Dimensiones finales: ";
    attended_cuda.getRowsCols();
    
    std::cout << "\n=== Resumen de Performance ===" << std::endl;
    std::cout << "Embedding Speedup: " << (float)cpu_time.count() / cuda_time.count() << "x" << std::endl;
    std::cout << "Attention Speedup: " << (float)cpu_time.count() / cuda_time.count() << "x" << std::endl;
    
    return 0;
}