#pragma once
#include "tensor.h"

// C[M, N] = A[M, K] @ B[K, N]
void matmul(const Tensor& A, const Tensor& B, Tensor& C);

// A += B, element-wise, shape must be same
void add_inplace(Tensor& A, const Tensor& B);

// 沿最后一维做softmax,原地
void softmax_inplace(Tensor& x);