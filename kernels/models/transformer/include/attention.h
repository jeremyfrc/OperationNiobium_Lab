#pragma once
#include "tensor.h"
#include "config.h"

struct AttentionWeights{
    Tensor wq; //[d_model, n_heads * d_head]
    Tensor Wk; //[d_model, n_kv_heads * d_head]
    Tensor Wv;  //[d_model, n_kv_heads * d_head]
    Tensor Wo;  //[n_heads * d_head, d_model]
};

// x: [seq_len, d_model] -> out: [seq_len, d_model]
// 内部： Q/K/V 投影 --> RoPE(Q,K) -> GQA 分组重复K/V --> casual mask 
// -> softmax(QK^T/ sqrt(d_head)) --> @V --> Wo 投影
void attention_forward(const Tensor& x, const AttentionWeights& w, const TransformerConfig& cfg, Tensor& out);