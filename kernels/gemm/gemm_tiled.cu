#include <iostream>
#include "gemm_utils.h"
#include <cuda_runtime.h>

#define TILE_DIM 16

__global__ void matmul_tiled(float* A, float* B, float* C, int N, int K, int M) {
    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_DIM + ty;
    int col = blockIdx.x * TILE_DIM + tx;
    float sum = 0.0f;
    for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; ++t) {

        if (row < N && t * TILE_DIM + tx < K) {
            // K是传进来的(一行几个元素)，row是行号
            // t * TILE_DIM + tx是当前线程在当前tile中对应的列索引
            As[ty][tx] = A[row * K + t * TILE_DIM + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if (t * TILE_DIM + ty < K && col < M) {
            // t * TILE_DIM + ty是在定位第几行
            // col是第几列，M是每列有多少个元素
            Bs[ty][tx] = B[(t * TILE_DIM + ty) * M + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        __syncthreads();
        for (int k = 0; k < TILE_DIM; ++k){
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }
    C[row * N + col]= sum;
}



int main() {

    int M = 1024, K = 1024, N = 1024;
    std::cout << "Matrix Matrix Multiplication (Tiled) Size: " << M << "x" << K << "x" << N << std::endl;

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // 分配 Host 内存
    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C_gpu = new float[M * N];
    float *h_C_cpu = new float[M * N];

    // 使用 helper 初始化数据
    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);

    // 分配 Device 内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    // 拷贝至 Device
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // 配置执行配置
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);

    // 启动 Kernel 并同步
    matmul_tiled<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    cudaDeviceSynchronize();

    // 拷回结果
    cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost);

    // CPU 计算验证
    std::cout << "Running CPU baseline for verification..." << std::endl;
    matmul_cpu(h_A, h_B, h_C_cpu, M, K, N);
    
    // 验证
    verify_gemm(h_C_cpu, h_C_gpu, M, N);

    // 清理
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_C_gpu; delete[] h_C_cpu;

    return 0;
}
