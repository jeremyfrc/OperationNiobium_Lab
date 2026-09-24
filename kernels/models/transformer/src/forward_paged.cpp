#include "transformer.h"
#include "attention_paged.h"
#include "rmsnorm.h"
#include "attention.h"
#include "swiglu.h"
#include "ops.h"
#include <limits>
#include <vector>
#include <cassert>
#include <algorithm>
#include "utils.h"


bool decode_layer_forward_paged(const Tensor& x, const DecoderLayerWeights& w, const TransformerConfig& cfg, Tensor& out, paged_kv::PagedKVCache& cache, paged_kv::BlockTable& table, int layer_idx, int pos_offset) {
    int seq_len = x.shape()[0];
    int d_model = cfg.d_model;

    // 1. RMSNorm
    Tensor x_norm1({seq_len, d_model});
    rmsnorm(x, w.rms1_weight, x_norm1, cfg.rms_eps);

    // 2. paged attention
    Tensor attn_out({seq_len, d_model});
    if (!attention_paged_forward(x_norm1, w.attn, cfg, attn_out, cache, table, layer_idx, pos_offset)) return false;

    // 3. 残差1
    std::copy(x.data(), x.data()+x.numel(), out.data());
    add_inplace(out, attn_out);

    // 4. 残差 RMSNorm
    Tensor x_norm2({seq_len, d_model});
    rmsnorm(out, w.rms2_weight, x_norm2, cfg.rms_eps);

    // 5. FFN
    Tensor ffn_out({seq_len, d_model});
    swiglu_forward(x_norm2, w.ffn, ffn_out);
    
    // 6. 残差2
    add_inplace(out, ffn_out);

    return true;
}

bool forward_paged(const std::vector<int>& token_ids, const TransformerWeights& w, const TransformerConfig& cfg, Tensor& logits, paged_kv::PagedKVCache& cache, paged_kv::BlockTable& table, int pos_offset) {
    int seq_len = token_ids.size();
    int d_model = cfg.d_model;

    Tensor x({seq_len, d_model});
    for(int i = 0; i < seq_len; ++i){
        int t = token_ids[i];
        std::copy(w.token_embedding.data() + t * d_model, w.token_embedding.data() + (t + 1) * d_model, x.data() + i * d_model);
    }

    // 缓冲
    Tensor buf_a({seq_len, d_model});
    Tensor buf_b({seq_len, d_model});
    std::copy(x.data(), x.data() + x.numel(), buf_a.data());
    Tensor* in = &buf_a;
    Tensor* outp = &buf_b;

    if (!paged_kv::prepare_sequence(table, seq_len)) return false;
    
    for (int l = 0; l < cfg.n_layers; ++l) {
        if (!decode_layer_forward_paged(*in, w.layers[l], cfg, *outp, cache, table, l, pos_offset)) return false;
        std::swap(in, outp);
    }

    // final rmsnorm + lm head
    Tensor x_final_norm({seq_len, d_model});
    rmsnorm(*in, w.final_rms_weight, x_final_norm, cfg.rms_eps);
    matmul(x_final_norm, w.lm_head, logits);

    return true;
}