#include <iostream>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>

// ---------------------------------------------------------
// [x] kernel: matmul_naive
// [x] 每 thread 负责一个 C[row][col]，内层 for k 循环累加
// ---------------------------------------------------------
__global__ void matmul_naive(float* A, float* B, float* C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float value = 0.0f;
        for (int k = 0; k < K; ++k) {
            value += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = value;
    }
}

// ---------------------------------------------------------
// [x] CPU baseline 验证
// ---------------------------------------------------------
void matmul_cpu(float* A, float* B, float* C, int M, int K, int N) {
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

// ---------------------------------------------------------
// [x] Correctness check (1e-4 tolerance)
// ---------------------------------------------------------
bool verify_result(float* cpu_res, float* gpu_res, int size) {
    for (int i = 0; i < size; ++i) {
        // 使用相对误差或绝对误差，1e-4 是深度学习单精度 float 常用的容忍度
        if (std::abs(cpu_res[i] - gpu_res[i]) > 1e-4) {
            printf("Mismatch at index %d: CPU=%f, GPU=%f\n", i, cpu_res[i], gpu_res[i]);
            return false;
        }
    }
    return true;
}

int main() {
    // 定义矩阵尺寸
    int N = 1024;
    size_t bytes = N * N * sizeof(float);

    // 1. Host 端：分配内存并初始化 (简化起见，全设为 1.0)
    float *h_A = new float[N * N];
    float *h_B = new float[N * N];
    float *h_C_cpu = new float[N * N];
    float *h_C_gpu = new float[N * N];
    for(int i = 0; i < N * N; ++i) { h_A[i] = 1.0f; h_B[i] = 1.0f; }

    // 2. Device 端：cudaMalloc / memcpy
    float *d_A, *d_B, *d_C;
    cudaMalloc(((void**)&d_A), bytes);
    cudaMalloc(((void**)&d_B), bytes);
    cudaMalloc(((void**)&d_C), bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // 3. [x] launch: 16x16 block，grid 覆盖整个 NxN 输出
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (N + 15) / 16); // 向上取整的工业级写法
    
    matmul_naive<<<grid, block>>>(d_A, d_B, d_C, N, N, N);
    cudaDeviceSynchronize(); // 确保 GPU 计算完成

    // 4. 拷回结果并运行 CPU baseline 进行验证
    cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost);
    matmul_cpu(h_A, h_B, h_C_cpu, N, N, N);

    if (verify_result(h_C_cpu, h_C_gpu, N * N)) {
        printf("Success! GPU results match CPU baseline.\n");
    }

    // 释放内存...
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C_cpu;
    delete[] h_C_gpu;
    return 0;
}