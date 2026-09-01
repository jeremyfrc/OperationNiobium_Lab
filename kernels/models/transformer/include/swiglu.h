#pragma once
#include "tensor.h"


struct FFNWeights{
    Tensor wGate;
    Tensor wUp;
    Tensor wDown;

    FFNWeights(Tensor gate, Tensor up, Tensor down) : wGate(std::move(gate)), wUp(std::move(up)), wDown(std::move(down)) {}
};

// 完整 FFN SwiGLU 前向算子:
// out = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down
void swiglu_forward(const Tensor& x, const FFNWeights& w, Tensor& out);