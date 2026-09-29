#pragma once
#include "tensor.h"
#include "config.h"
#include "attention.h"        // AttentionWeights
#include "paged_kv_cache.h"
#include "block_table.h"
#include "sequence_manager.h"

// paged 路径的 attention 入口: 投影 → RoPE → cache.write → gather → attention_over_kv → wO
void attention_paged_forward(const Tensor& x, const AttentionWeights& w, const TransformerConfig& cfg, Tensor& out,
                             paged_kv::PagedKVCache& cache, paged_kv::BlockTable& table,
                             int layer_idx, int pos_offset);