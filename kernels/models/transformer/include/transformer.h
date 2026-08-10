#pragma once
include <vector>
#include "tensor.h"
#include "config.h"
#include "attention"
#include "swiglu.h"

struct DecoderLayerWeights{
    Tensor rms1_weight;
    AttentionWeights attn;
    Tensor rms2_weight;
    FFNWeights ffn;
};

struct TransformerWeights{
    Tensor token_embedding;   // [vocab_size, d_model]
    std::vector<DecoderLayerWeights> layers;
    Tensor final_rms_weight;
    Tensor lm_head;          // [d_model, vocab_size]
};

// 单层前向: x --> out , shape均[seq_len, d_model]
void decoder_layer_forward(const Tensor& x, const DecoderLayerWeights& w, const TransformerConfig& cfg, Tensor& out);

// 整栈前向: token_ids -> logits [seq_len, vocab_size]
void transformer_forward(const std::vector<int>& token_ids, const TransformerWeights& w, const TransformerConfig& cfg, Tensor& logits);

// 自回归生成: greedy argmax版本先实现，采样可选stretch
std::vector<int> generate(const std::vector<int>& prompt_ids, const TransformerWeights& w, const TransformerConfig& cfg, int max_new_tokens);