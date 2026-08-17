#include <cassert>
#include <cmath>
#include <algorithm>
#include "tensor.h"
#include "ops.h"


void add_inplace(Tensor& A, const Tensor& B){
    assert(A.numel() == B.numel() && "Tensor sizes must match for add_inplace!");
    float* a_ptr = A.data();
    const float* b_ptr = B.data();

    int size = A.numel();
    for (int i = 0; i < size; ++i){
        a_ptr[i] += b_ptr[i];
    }
}

void softmax_inplace(Tensor& x){
    const std::vector<int>& shape = x.shape();
    assert(!shape.empty() && "Tensor shape cannot be empty!");
    int N = shape.back();

    int totalElements = x.numel();
    int numRows = totalElements / N;

    float* dataPtr = x.data();

    for (int r = 0; r < numRows; ++r){
        float* row = dataPtr + r * N;

        float maxVal = row[0];
        for (int j = 1; j < N; ++j){
            if (row[j] > maxVal) {
                maxVal = row[j];
            }
        }

        float sumExp = 0.0f;
        for (int j = 0; j < N; ++j) {
            row[j] = std::exp(row[j] - maxVal);
            sumExp += row[j];
        }

        float invSum = 1.0 / sumExp;
        for (int j = 0; j < N; ++j){
            row[j] *= invSum;
        }
    }

}


void matmul(const Tensor& A, const Tensor& B, Tensor& C){
    int M = A.shape()[0];
    int K = A.shape()[1];
    int N = B.shape()[1];

    assert(A.shape()[1] == B.shape()[0] && "A's cols must be same as B's rows!");
    assert(C.shape()[0] == M && C.shape()[1] == N && "C shape mismatch!");

    const float* a_ptr = A.data();
    const float* b_ptr = B.data();
    float* c_ptr = C.data();

    std::fill(c_ptr, c_ptr + M * N, 0.0f);

    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k){
            float aVal = a_ptr[i * K + k];
            for (int j = 0; j < N; ++j){
                c_ptr[i * N + j] += aVal * b_ptr[k * N + j];
            }
        }
    }
}