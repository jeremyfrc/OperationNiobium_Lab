#include "swiglu.h"
#include "ops.h"
#include <cmath>
#include <cassert>

// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
static inline float silu(float x){
    return x / (1.0f + std::exp(-x));
}

// void swiglu_elementwise(const Tensor& gate, const Tensor& up, Tensor& out) {
//     if (gate.numel() != up.numel() || out.numel() != gate.numel()) {
//         throw std::runtime_error("SwiGLU elementwise: Tensor size mismatch!");
//     }

//     const float* gPtr = gate.data();
//     const float* uPtr = up.data();
//     float* outPtr = out.data();
//     size_t total_elements = gate.numel();

//     for (size_t i = 0; i < total_elements; ++i) {
//         outPtr[i] = silu(gPtr[i]) * uPtr[i];
//     }
// }

void swiglu_forward(const Tensor& x, const FFNWeights& w, Tensor& out) {
    // w.wGate 形状为 [in_dim, intermediate_dim]
    int in_dim = w.wGate.shape()[0];
    int intermediate_dim = w.wGate.shape()[1];
    
    // token 总数 = 全部元素量 / 输入维度
    int num_tokens = x.numel() / in_dim; 

    Tensor gateProj({num_tokens, intermediate_dim});
    Tensor upProj({num_tokens, intermediate_dim});

    // x: [num_tokens, in_dim], wGate: [in_dim, intermediate_dim]
    matmul(x, w.wGate, gateProj); 
    matmul(x, w.wUp, upProj);

    Tensor intermediate({num_tokens, intermediate_dim});
    const float* gPtr = gateProj.data();
    const float* uPtr = upProj.data();
    float* interPtr = intermediate.data();

    int total_elements = intermediate.numel();
    for (int i = 0; i < total_elements; ++i) {
        interPtr[i] = silu(gPtr[i]) * uPtr[i];
    }

    // intermediate: [num_tokens, intermediate_dim], wDown: [intermediate_dim, in_dim]
    matmul(intermediate, w.wDown, out);
}
