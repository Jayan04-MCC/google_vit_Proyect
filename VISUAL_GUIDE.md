# Vision Transformer (ViT) - Guía Visual

## Arquitectura General

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           VISION TRANSFORMER PIPELINE                       │
└─────────────────────────────────────────────────────────────────────────────┘

INPUT IMAGE (224x224x3)
         │
         ▼
┌─────────────────────────┐
│   PATCH EXTRACTION      │    Divide imagen en patches de 16x16
│   (14x14 patches)       │    → 196 patches de 768 dimensiones
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│   PATCH EMBEDDING       │    Transforma patches usando pesos pre-entrenados
│   Linear Projection     │    196x768 → 196x768
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│   ADD CLS TOKEN +       │    Agrega token de clasificación + embeddings
│   POSITION EMBEDDINGS   │    posicionales: 196x768 → 197x768
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│   MULTI-HEAD ATTENTION  │    Procesa secuencia con 12 cabezas de atención
│   (12 heads)            │    197x768 → 197x768
└─────────────────────────┘
```

## Flujo de Datos Detallado

### 1. Patch Extraction & Embedding
```
Original Image                    Patches                    Embedded Patches
┌─────────────────┐              ┌─────────────────┐        ┌─────────────────┐
│ 224x224x3       │    split     │ 196 patches     │ embed  │ 196 x 768       │
│                 │ ──────────►  │ each 16x16x3    │ ────► │                 │
│     [IMAGE]     │              │ → flattened     │        │   [VECTORS]     │
└─────────────────┘              └─────────────────┘        └─────────────────┘
```

### 2. Input Construction
```
Embedded Patches         CLS Token           Position Embeddings         Final Input
┌─────────────────┐     ┌─────────┐         ┌─────────────────┐         ┌─────────────────┐
│ 196 x 768       │  +  │ 1 x 768 │    +    │ 197 x 768       │    =    │ 197 x 768       │
│                 │     │         │         │                 │         │                 │
│   [PATCHES]     │     │ [CLS]   │         │   [POSITIONS]   │         │   [FINAL]       │
└─────────────────┘     └─────────┘         └─────────────────┘         └─────────────────┘
```

### 3. Multi-Head Self-Attention
```
Input Sequence (197 x 768)
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MULTI-HEAD ATTENTION                                │
│                                                                             │
│  Query (Q)           Key (K)            Value (V)                          │
│  197x768 ────►      197x768 ────►      197x768 ────►                       │
│      │                 │                 │                                 │
│      ▼                 ▼                 ▼                                 │
│  ┌─────────┐      ┌─────────┐      ┌─────────┐                             │
│  │ Head 1  │      │ Head 1  │      │ Head 1  │                             │
│  │ Head 2  │      │ Head 2  │      │ Head 2  │                             │
│  │   ...   │      │   ...   │      │   ...   │                             │
│  │ Head 12 │      │ Head 12 │      │ Head 12 │                             │
│  └─────────┘      └─────────┘      └─────────┘                             │
│         │              │              │                                   │
│         └──────────────┼──────────────┘                                   │
│                        ▼                                                   │
│                 Attention(Q,K,V)                                           │
│                        │                                                   │
│                        ▼                                                   │
│                 Concatenate Heads                                          │
│                        │                                                   │
│                        ▼                                                   │
│                 Linear Projection                                          │
└─────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
Output (197 x 768)
```

## Dimensiones en cada paso

| Paso | Entrada | Salida | Operación |
|------|---------|---------|-----------|
| **Patch Extraction** | 224×224×3 | 196×768 | Divide en patches 16×16, flatten |
| **Patch Embedding** | 196×768 | 196×768 | Proyección lineal |
| **Add CLS + Position** | 196×768 | 197×768 | Concatena CLS token + suma posiciones |
| **Multi-Head Attention** | 197×768 | 197×768 | Atención con 12 cabezas |

## Código correspondiente

```cpp
// main.cpp - Flujo principal

// 1. Carga patches (196x768)
Matrix patches = Matrix::fromCSV("patches_00001.csv", false);

// 2. Embedding de patches
PatchEmbedder embedder(weight_file, bias_file);
Matrix embedded = embedder.embed(patches);  // 196x768

// 3. Construcción de entrada con CLS + posiciones
InputBuilder builder(cls_token_file, position_embeddings_file);
Matrix input = builder.build(embedded);  // 197x768

// 4. Atención multi-cabeza
MultiHeadSelfAttention attention(query_weights, key_weights, value_weights, ...);
Matrix attended = attention.forward(input);  // 197x768
```

## Archivos de pesos utilizados

```
pesos/
├── vit_embeddings_patch_embeddings_projection_weight.csv    # Pesos embedding patches
├── vit_embeddings_patch_embeddings_projection_bias.csv      # Bias embedding patches
├── vit_embeddings_cls_token.csv                             # Token CLS
├── vit_embeddings_position_embeddingsV2.csv                 # Embeddings posicionales
├── vit_encoder_layer_0_attention_attention_query_weight.csv # Pesos Query
├── vit_encoder_layer_0_attention_attention_query_bias.csv   # Bias Query
├── vit_encoder_layer_0_attention_attention_key_weight.csv   # Pesos Key
├── vit_encoder_layer_0_attention_attention_key_bias.csv     # Bias Key
├── vit_encoder_layer_0_attention_attention_value_weight.csv # Pesos Value
├── vit_encoder_layer_0_attention_attention_value_bias.csv   # Bias Value
├── vit_encoder_layer_0_attention_output_dense_weight.csv    # Pesos salida
└── vit_encoder_layer_0_attention_output_dense_bias.csv      # Bias salida
```

## Notas importantes

- **ViT Base**: 12 capas, 12 cabezas de atención, 768 dimensiones
- **Patches**: 16×16 píxeles cada uno
- **Secuencia**: 197 tokens (196 patches + 1 CLS token)
- **Atención**: Cada cabeza procesa 64 dimensiones (768/12 = 64)