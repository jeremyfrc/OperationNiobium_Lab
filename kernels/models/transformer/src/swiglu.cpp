#include "swiglu.h"
#include "ops.h"
#include <cmath>
#include <cassert>

// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
static inline float silu(float x){
    return x / (1.0f + std::exp(-x));
}

void swiglu_forward(const Tensor& x, const FFNWeights& w, Tensor& out) {
    // 假设 x shape: [seq_len, hidden_dim]
    // w_gate shape: [hidden_dim, intermediate_dim]
    // w_up shape:   [hidden_dim, intermediate_dim]
    // w_down shape: [intermediate_dim, hidden_dim]

    // 1. 计算 gate_proj = x @ W_gate  -> shape: [seq_len, intermediate_dim]
    //    计算 up_proj   = x @ W_up    -> shape: [seq_len, intermediate_dim]
    int seq_len = x.shape()[0];
    int intermediate_dim = w.wGate.shape()[1];

    Tensor gateProj({seq_len, intermediate_dim});
    Tensor upProj({seq_len, intermediate_dim});

    matmul(x, w.wGate, gateProj);
    matmul(x, w.wUp, upProj);

    Tensor intermediate({seq_len, intermediate_dim});
    const float* gPtr = gateProj.data();
    const float* uPtr = upProj.data();
    float* interPtr = intermediate.data();

    int total_elements = intermediate.numel();
    for(int i = 0; i< total_elements; ++i){
        interPtr[i] = silu(gPtr[i]) * uPtr[i];
    }

    matmul(intermediate, w.wDown, out);
}