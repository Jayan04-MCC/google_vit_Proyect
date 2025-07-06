#include <iostream>
#include "include/matrix.h"
#include "include/PatchEmbedder.h"
#include "include/InputBuilder.h"
#include "include/MultiHeadSelfAttention.h"
int main() {
    Matrix patches = Matrix::fromCSV("C:/Users/JAYAN/CLionProjects/google-vit/patches_00001.csv",false); // (196 x 768)

    PatchEmbedder embedder(
        "C:/Users/JAYAN/CLionProjects/google-vit/pesos/vit_embeddings_patch_embeddings_projection_weight.csv",
        "C:/Users/JAYAN/CLionProjects/google-vit/pesos/vit_embeddings_patch_embeddings_projection_bias.csv"
    );

    Matrix embedded = embedder.embed(patches);  // (196 x 768)

    embedded.print(1, 5);  // Muestra 1ra fila, 5 columnas
    std::cout << "Filas y columnas de embedded: "<<std::endl;
    embedded.getRowsCols();
    InputBuilder builder(
        "C:/Users/JAYAN/CLionProjects/google-vit/pesos/vit_embeddings_cls_token.csv",
        "C:/Users/JAYAN/CLionProjects/google-vit/pesos/vit_embeddings_position_embeddingsV2.csv"
    );
    Matrix input = builder.build(embedded);  // 197 x 768

    input.print(1, 5);  // imprime primera fila con 5 columnas
    // Paso 4: Aplicar Atención (Layer 0)
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

    Matrix attended = attention.forward(input);  // (197 x 768)

    attended.print(1, 5);  // muestra la salida de la atención
    std::cout << "Filas y columnas de attended:\n";
    attended.getRowsCols();

    return 0;
}
