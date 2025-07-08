#include <iostream>
#include "include/matrix.h"
#include "include/PatchEmbedder.h"
#include "include/InputBuilder.h"
#include "include/TransformerEncoderLayer.h"

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

    Matrix x = input;
    for (int i = 0; i < 12; ++i) {
        std::string prefix = "C:/Users/JAYAN/CLionProjects/google-vit/pesos/vit_encoder_layer_" + std::to_string(i);

        TransformerEncoderLayer layer(
            prefix + "_layernorm_before_weight.csv",
            prefix + "_layernorm_before_bias.csv",
            prefix + "_attention_attention_query_weight.csv",
            prefix + "_attention_attention_query_bias.csv",
            prefix + "_attention_attention_key_weight.csv",
            prefix + "_attention_attention_key_bias.csv",
            prefix + "_attention_attention_value_weight.csv",
            prefix + "_attention_attention_value_bias.csv",
            prefix + "_attention_output_dense_weight.csv",
            prefix + "_attention_output_dense_bias.csv",

            prefix + "_layernorm_after_weight.csv",
            prefix + "_layernorm_after_bias.csv",
            prefix + "_intermediate_dense_weight.csv",
            prefix + "_intermediate_dense_bias.csv",

            prefix + "_output_dense_weight.csv",
            prefix + "_output_dense_bias.csv",12
        );

        std::cout << "Procesando capa " << i << std::endl;
        x = layer.forward(x);  // (197 x 768)
    }

    // 4. Extraer el embedding del token CLS (fila 0)
    Matrix cls_embedding = x.sliceRows(0, 1);  // (1 x 768)

    // Mostrar resultado
    cls_embedding.print(1, 10);  // Muestra primera fila con 10 columnas
    std::cout << "Proceso completo de forward del ViT.\n";

    return 0;
}
