#include "rope.h"
#include <cmath>
#include <cassert>

void rope_inplace(Tensor& x, int numHeads, int posOffset, float base){
    const auto& shape = x.shape();
    assert(!shape.empty() && "Tensor shape cannot be empty for RoPE!");

    int headDim = shape.back();
    assert(headDim % 2 == 0 && "headDim must be even for RoPE!");

    int totalElements = x.numel();
    int numRows = totalElements / headDim;

    float* xPtr = x.data();

    for (int r = 0; r < numRows; ++r){
        int seqIdx = (r / numHeads) + posOffset;
        float* xRow = xPtr + r * headDim;

        for (int i = 0; i < headDim/2; ++i){
            float freq = 1.0f / std::pow(base, static_cast<float>(2 * i) / static_cast<float>(headDim));
            float val = static_cast<float>(seqIdx) * freq;

            float cosVal = std::cos(val);
            float sinVal = std::sin(val);

            float x0 = xRow[2 * i];
            float x1 = xRow[2 * i + 1];

            xRow[2*i] = x0 * cosVal - x1 * sinVal;
            xRow[2*i+1] = x0 * sinVal + x1 * cosVal;
        }
    }
}


void rope(const Tensor& x, Tensor& out, int numHeads, int posOffset, float base){
    assert(out.numel() == x.numel() && "Output tensor size must match input tensor size!");
    std::copy(x.data(), x.data()+x.numel(), out.data());
    rope_inplace(out, numHeads, posOffset, base);
}