#include "rmsnorm.h"
#include <cmath>
#include <cassert>

void rmsnorm(const Tensor& x, const Tensor& weight, Tensor& out, float eps){
    const auto& shape = x.shape();
    assert(!shape.empty() && "Tensor shape cannot be empty for rmsnorm!");

    int N = shape.back();
    int numRows = x.numel() / N;

    assert(weight.numel() == static_cast<size_t>(N) && "Weiht size must match last dimension of input!");
    assert(out.numel() == x.numel() && "Output tensor size must match input tensor size!");

    const float* xPtr = x.data();
    const float* wPtr = weight.data();
    float* outPtr = out.data();

    for (int r = 0; r < numRows; ++r) {
        const float* xRow = xPtr + r * N;
        float* outRow = outPtr + r * N;

        float sumSq = 0.0f;
        for (int j = 0; j < N; ++j) {
            sumSq += xRow[j] * xRow[j];
        }

        float meanSq = sumSq / static_cast<float>(N);
        float invRms = 1.0f / std::sqrt(meanSq + eps);

        for (int j = 0; j < N; ++j) {
            outRow[j] = xRow[j] * invRms * wPtr[j];
        }
    }
}