#include "rope.h"
#include "check.h"
#include <cmath>

void rope_inplace(Tensor& x, int numHeads, int headDim, int posOffset, float base){
    const auto& shape = x.shape();
    
    // 契约： head_dim显示传入，且必须等于张量最后一维。
    NB_CHECK(!shape.empty(), "rope: Tensor shape cannot be empty");
    NB_CHECK(shape.back() == headDim, "rope: last dim must equal head_dim (did you pass [seq, heads*head_dim] ?)");
    NB_CHECK(headDim % 2 == 0, "rope: headDim must be even.");
    NB_CHECK(x.numel() % (numHeads * headDim) == 0, "rope: numel not divisible by numHeads*head_dim");

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


void rope(const Tensor& x, Tensor& out, int numHeads, int headDim, int posOffset, float base){
    NB_CHECK(out.numel() == x.numel(), "rope: output tensor size must match input tensor size");
    std::copy(x.data(), x.data()+x.numel(), out.data());
    rope_inplace(out, numHeads, headDim, posOffset, base);
}