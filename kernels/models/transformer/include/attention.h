#pragma once
#include "tensor.h"
#include "config.h"
#include "kv_cache.h"

struct AttentionWeights{
    Tensor wQ; //[d_model, n_heads * d_head]
    Tensor wK; //[d_model, n_kv_heads * d_head]
    Tensor wV;  //[d_model, n_kv_heads * d_head]
    Tensor wO;  //[n_heads * d_head, d_model]
};

// x: [seq_len, d_model] -> out: [seq_len, d_model]
// 内部： Q/K/V 投影 --> RoPE(Q,K) -> GQA 分组重复K/V --> casual mask 
// -> softmax(QK^T/ sqrt(d_head)) --> @V --> Wo 投影
void attention_forward(const Tensor& x, const AttentionWeights& w, const TransformerConfig& cfg,Tensor& out);

void attention_forward_kv(const Tensor& x, const AttentionWeights& w, const TransformerConfig& cfg, Tensor& out, KVCache& kv_cache, int layer_idx, int pos_offset);