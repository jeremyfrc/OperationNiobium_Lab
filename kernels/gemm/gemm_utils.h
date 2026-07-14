// kernels/gemm/gemm_utils.h
#pragma once
#include <iostream>
#include <cmath>
#include <cstdlib>

// 1. 矩阵初始化：填充 0.0f 到 1.0f 之间的随机数
inline void init_matrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

// 2. CPU 矩阵乘法基准（用于验证正确性）
inline void matmul_cpu(const float* A, const float* B, float* C, int M, int K, int N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// 3. 结果验证函数
inline bool verify_gemm(const float* cpu_res, const float* gpu_res, int M, int N) {
    const float epsilon = 1e-4f;
    for (int i = 0; i < M * N; ++i) {
        if (std::fabs(cpu_res[i] - gpu_res[i]) > epsilon) {
            std::cout << "Verification FAILED at index " << i 
                      << " (CPU: " << cpu_res[i] << ", GPU: " << gpu_res[i] << ")\n";
            return false;
        }
    }
    std::cout << "Verification SUCCESS! GPU results match CPU baseline.\n";
    return true;
}