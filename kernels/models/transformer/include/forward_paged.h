#pragma once
#include "transformer.h"        // DecoderLayerWeights, TransformerWeights, TransformerConfig
#include "tensor.h"
#include "attention_paged.h"    // attention_paged_forward
#include "paged_kv_cache.h"
#include "block_table.h"


// 单层 (paged): 与 decode_layer_forward_kv 孪生, 只把 attention 换成 paged 版
// 返回 false: attention_paged_forward 内部 append_kv OOM
bool decode_layer_forward_paged(const Tensor& x, const DecoderLayerWeights& w, const TransformerConfig& cfg, Tensor& out, paged_kv::PagedKVCache& cache, paged_kv::BlockTable& table, int layer_idx, int pos_offset);

// 整栈 (paged): 与 transformer_forward_kv 孪生
bool forward_paged(const std::vector<int>& token_ids, const TransformerWeights& w, const TransformerConfig& cfg, Tensor& logits, paged_kv::PagedKVCache& cache, paged_kv::BlockTable& table, int pos_offset);