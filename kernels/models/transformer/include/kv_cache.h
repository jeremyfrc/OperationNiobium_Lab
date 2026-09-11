#pragma once
#include "tensor.h"
#include <vector>

struct KVCache{
    std::vector<Tensor> k_caches; // [n_layers] -> [max_seq_len, n_kv_heads, d_head]
    std::vector<Tensor> v_caches;

    // ctor
    KVCache(int n_layers, int max_seq_len, int n_kv_heads, int d_head){
        for (int i = 0; i < n_layers; ++i){
            k_caches.emplace_back(std::vector<int>{max_seq_len, n_kv_heads, d_head});
            v_caches.emplace_back(std::vector<int>{max_seq_len, n_kv_heads, d_head});
        }
    }
};