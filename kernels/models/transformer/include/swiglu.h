#pragma once
#include "tensor.h"


struct FFNWeights{
    Tensor wGate;
    Tensor wUp;
    Tensor wDown;

    FFNWeights(Tensor gate, Tensor up, Tensor down) : wGate(std::move(gate)), wUp(std::move(up)), wDown(std::move(down)) {}
};

// // 1. 纯元素级激活算子 (Stage 1 单测直接测这个)
// void swiglu_elementwise(const Tensor& gate, const Tensor& up, Tensor& out);

// 2. 完整 FFN 前向算子 (Stage 3 整层/整栈使用)
void swiglu_forward(const Tensor& x, const FFNWeights& w, Tensor& out);