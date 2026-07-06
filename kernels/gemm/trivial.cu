#include <cuda_runtime.h>

__global__ void vec_add(float* a, float* b, float* c, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main(){
    int n = 1 << 20;
    float *da, *db, *dc;
    cudaMalloc(&da, n*sizeof(float));
    cudaMalloc(&db, n*sizeof(float));
    cudaMalloc(&dc, n*sizeof(float));
    vec_add<<<(n+255)/256, 256>>>(da, db, dc, n);
    cudaDeviceSynchronize();
}
