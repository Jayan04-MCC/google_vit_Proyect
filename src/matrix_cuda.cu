#include "../include/matrix.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>

// Kernel básico para multiplicación elemento por elemento
__global__ void elementwise_add_kernel(float* a, float* b, float* result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] + b[idx];
    }
}

// Kernel básico para multiplicación por escalar
__global__ void scalar_multiply_kernel(float* data, float scalar, float* result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = data[idx] * scalar;
    }
}

// Kernel básico para multiplicación de matrices (versión simple)
__global__ void matrix_multiply_kernel(float* a, float* b, float* c, 
                                     int rows_a, int cols_a, int cols_b) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows_a && col < cols_b) {
        float sum = 0.0f;
        for (int k = 0; k < cols_a; ++k) {
            sum += a[row * cols_a + k] * b[k * cols_b + col];
        }
        c[row * cols_b + col] = sum;
    }
}

// Función para suma con CUDA
Matrix Matrix::cuda_add(const Matrix& other) const {
    // Verificar dimensiones
    if (rows != other.rows || cols != other.cols) {
        throw std::runtime_error("Dimensiones incompatibles para suma CUDA");
    }
    
    int size = rows * cols;
    size_t bytes = size * sizeof(float);
    
    // Alocar memoria en GPU
    float *d_a, *d_b, *d_result;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_result, bytes);
    
    // Copiar datos a GPU
    cudaMemcpy(d_a, data.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, other.data.data(), bytes, cudaMemcpyHostToDevice);
    
    // Configurar grid y bloques
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    
    // Ejecutar kernel
    elementwise_add_kernel<<<gridSize, blockSize>>>(d_a, d_b, d_result, size);
    
    // Verificar errores
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Error CUDA: " << cudaGetErrorString(err) << std::endl;
    }
    
    // Crear matriz resultado y copiar datos de vuelta
    Matrix result(rows, cols);
    cudaMemcpy(result.data.data(), d_result, bytes, cudaMemcpyDeviceToHost);
    
    // Liberar memoria GPU
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
    
    return result;
}

// Función para multiplicación por escalar con CUDA
Matrix Matrix::cuda_scalar_multiply(float scalar) const {
    int size = rows * cols;
    size_t bytes = size * sizeof(float);
    
    float *d_data, *d_result;
    cudaMalloc(&d_data, bytes);
    cudaMalloc(&d_result, bytes);
    
    cudaMemcpy(d_data, data.data(), bytes, cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    
    scalar_multiply_kernel<<<gridSize, blockSize>>>(d_data, scalar, d_result, size);
    
    Matrix result(rows, cols);
    cudaMemcpy(result.data.data(), d_result, bytes, cudaMemcpyDeviceToHost);
    
    cudaFree(d_data);
    cudaFree(d_result);
    
    return result;
}

// Función para multiplicación de matrices con CUDA
Matrix Matrix::cuda_multiply(const Matrix& other) const {
    if (cols != other.rows) {
        throw std::runtime_error("Dimensiones incompatibles para multiplicación CUDA");
    }
    
    size_t bytes_a = rows * cols * sizeof(float);
    size_t bytes_b = other.rows * other.cols * sizeof(float);
    size_t bytes_c = rows * other.cols * sizeof(float);
    
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes_a);
    cudaMalloc(&d_b, bytes_b);
    cudaMalloc(&d_c, bytes_c);
    
    cudaMemcpy(d_a, data.data(), bytes_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, other.data.data(), bytes_b, cudaMemcpyHostToDevice);
    
    // Configurar grid 2D para multiplicación de matrices
    dim3 blockSize(16, 16);
    dim3 gridSize((other.cols + blockSize.x - 1) / blockSize.x,
                  (rows + blockSize.y - 1) / blockSize.y);
    
    matrix_multiply_kernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, 
                                                   rows, cols, other.cols);
    
    Matrix result(rows, other.cols);
    cudaMemcpy(result.data.data(), d_c, bytes_c, cudaMemcpyDeviceToHost);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return result;
}

// Función optimizada con cuBLAS para multiplicación de matrices
Matrix Matrix::cublas_multiply(const Matrix& other) const {
    if (cols != other.rows) {
        throw std::runtime_error("Dimensiones incompatibles para multiplicación cuBLAS");
    }
    
    // Crear handle cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    size_t bytes_a = rows * cols * sizeof(float);
    size_t bytes_b = other.rows * other.cols * sizeof(float);
    size_t bytes_c = rows * other.cols * sizeof(float);
    
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes_a);
    cudaMalloc(&d_b, bytes_b);
    cudaMalloc(&d_c, bytes_c);
    
    cudaMemcpy(d_a, data.data(), bytes_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, other.data.data(), bytes_b, cudaMemcpyHostToDevice);
    
    // Parámetros para SGEMM
    const float alpha = 1.0f, beta = 0.0f;
    
    // cuBLAS usa column-major, necesitamos ajustar
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                other.cols, rows, cols,
                &alpha,
                d_b, other.cols,
                d_a, cols,
                &beta,
                d_c, other.cols);
    
    Matrix result(rows, other.cols);
    cudaMemcpy(result.data.data(), d_c, bytes_c, cudaMemcpyDeviceToHost);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cublasDestroy(handle);
    
    return result;
}