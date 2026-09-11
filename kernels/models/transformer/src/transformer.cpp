#include "transformer.h"
#include "rmsnorm.h"
#include "attention.h"
#include "swiglu.h"
#include "ops.h"
#include <limits>
#include <vector>
#include <cassert>
#include <algorithm>
#include "utils.h"


// 单层 Decoder Layer 前向: 
// x -> RMSNorm -> Attention -> Add -> RMSNorm -> SwiGLU -> Add -> out
void decoder_layer_forward(const Tensor& x,
                           const DecoderLayerWeights& w,
                           const TransformerConfig& cfg,
                           Tensor& out) {
    int seq_len = x.shape()[0];
    int d_model = cfg.d_model;

    // 1. Pre-attention RMSNorm
    Tensor x_norm1({seq_len, d_model});
    // 调用 rmsnorm(x, w.rms1_weight, cfg.rms_eps, x_norm1);
    rmsnorm(x, w.rms1_weight, x_norm1, cfg.rms_eps);

    // 2. Attention
    Tensor attn_out({seq_len, d_model});
    attention_forward(x_norm1, w.attn, cfg, attn_out);

    // 3. Residual Connection 1 (x = x + attn_out)
    // 注意：为了不破坏输入的 x（如果后面还要用），我们先拷贝一份或者直接输出到 out
    std::copy(x.data(), x.data() + x.numel(), out.data());
    // 调用 add_inplace(out, attn_out);
    add_inplace(out, attn_out);

    // 4. Pre-FFN RMSNorm
    Tensor x_norm2({seq_len, d_model});
    // 调用 rmsnorm(out, w.rms2_weight, cfg.rms_eps, x_norm2);
    rmsnorm(out, w.rms2_weight, x_norm2, cfg.rms_eps);
    // 5. Feed Forward (SwiGLU)
    Tensor ffn_out({seq_len, d_model});
    // 调用 swiglu_forward(x_norm2, w.ffn, ffn_out);
    swiglu_forward(x_norm2, w.ffn, ffn_out);

    // 6. Residual Connection 2 (out = out + ffn_out)
    // 调用 add_inplace(out, ffn_out);
    add_inplace(out, ffn_out);
}

// 整栈前向: token_ids -> Embedding -> Layers -> Final Norm -> LM Head -> Logits
void transformer_forward(const std::vector<int>& token_ids, const TransformerWeights& w, const TransformerConfig& cfg, Tensor& logits) {
    int seq_len = token_ids.size();
    int d_model = cfg.d_model;

    // 1. Embedding Lookup
    // 将 token_ids 转换为 [seq_len, d_model] 的 Tensor
    Tensor x({seq_len, d_model});
    
    /*
    for (int i = 0; i < seq_len; ++i) {
        int token_id = token_ids[i];
        // 从 w.token_embedding 中拷贝对应的行到 x 的第 i 行
    }
    */
    for (int i = 0; i < seq_len; ++i) {
        int token_id = token_ids[i];
        assert(token_id < cfg.vocab_size && "token_id out of bounds");
        // 从 w.token_embedding 中拷贝对应的行到 x 的第 i 行
        std::copy(w.token_embedding.data() + token_id * d_model,
                  w.token_embedding.data() + (token_id + 1) * d_model,
                  x.data() + i * d_model);
    }

    // 2. Decoder Layers
    // 首先将 Embedding 输出拷贝到 buffer_a
    Tensor buffer_a({seq_len, d_model});
    std::copy(x.data(), x.data() + x.numel(), buffer_a.data());
    Tensor buffer_b({seq_len, d_model});
    Tensor* input_ptr = &buffer_a;
    Tensor* output_ptr = &buffer_b;

    for (int l = 0; l < cfg.n_layers; ++l) {
        decoder_layer_forward(*input_ptr, w.layers[l], cfg, *output_ptr);
        // 更新 layer_in_out = current_layer_out，准备下一层
        std::swap(input_ptr, output_ptr);
    }

    // 3. Final RMSNorm
    Tensor x_final_norm({seq_len, d_model});
    // 调用 rmsnorm(layer_in_out, w.final_rms_weight, cfg.rms_eps, x_final_norm);
    rmsnorm(*input_ptr, w.final_rms_weight, x_final_norm, cfg.rms_eps);

    // 4. LM Head (Linear to Logits)
    // logits 维度应该是 [seq_len, vocab_size]
    // 调用 matmul(x_final_norm, w.lm_head, logits);
    matmul(x_final_norm, w.lm_head, logits);
}

// Stage 4 才会用到的生成逻辑，现在可以留空
std::vector<int> generate(const std::vector<int>& prompt_ids, const TransformerWeights& w, const TransformerConfig& cfg, int max_new_tokens) {
    std::vector<int> total_ids = prompt_ids;

    for (int i = 0; i < max_new_tokens; ++i) {
        int current_seq_len = total_ids.size();

        // 1. 创建用于存储输出的 logits Tensor [current_seq_len, vocab_size]
        Tensor logits({current_seq_len, cfg.vocab_size});

        // 2. 调用 Stage 3 已经写好的 forward
        transformer_forward(total_ids, w, cfg, logits);

        // 3. 取出最后一个 token 的 logits, 最后一个 token 的起始指针： logits.data() + (last_row_index * vocab_size)
        float* last_token_logits = logits.data() + (current_seq_len - 1) * cfg.vocab_size;

        // 4. Greedy search (argmax)
        int next_token_id = 0;
        float max_logit = last_token_logits[0];
        for (int v = 1; v < cfg.vocab_size; ++v) {
            if (last_token_logits[v] > max_logit) {
                max_logit = last_token_logits[v];
                next_token_id = v;
            }
        }

        // 5. 把它存入序列
        total_ids.push_back(next_token_id);
        //std::cout << "Generated token: " << next_token_id << std::endl;

        // if (next_token_id == cfg.eos_token_id) break;
    }
    return total_ids;
}


void decode_layer_forward_kv(const Tensor& x, const DecoderLayerWeights& w, const TransformerConfig& cfg, Tensor& out, KVCache& kv_cache, int layer_idx, int pos_offset) {
    int seq_len = x.shape()[0];
    int d_model = cfg.d_model;

    // 1. RMSNorm
    Tensor x_norm1({seq_len, d_model});
    rmsnorm(x, w.rms1_weight, x_norm1, cfg.rms_eps);

    // 2. KV-attention
    Tensor attn_out({seq_len, d_model});
    attention_forward_kv(x_norm1, w.attn, cfg, attn_out, kv_cache, layer_idx, pos_offset);

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
}

void transformer_forward_kv(const std::vector<int>& token_ids, const TransformerWeights& w, const TransformerConfig& cfg, Tensor& logits, KVCache& kv_cache, int pos_offset) {
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

    for (int l = 0; l < cfg.n_layers; ++l) {
        decode_layer_forward_kv(*in, w.layers[l], cfg, *outp, kv_cache, l, pos_offset);
        std::swap(in, outp);
    }

    // final rmsnorm + lm head
    Tensor x_final_norm({seq_len, d_model});
    rmsnorm(*in, w.final_rms_weight, x_final_norm, cfg.rms_eps);
    matmul(x_final_norm, w.lm_head, logits);
}


std::vector<int> generate_kv(const std::vector<int>& prompt_ids, const TransformerWeights& w, const TransformerConfig& cfg, int max_new_tokens) {
    int capacity = (int)prompt_ids.size() + max_new_tokens;
    KVCache cache(cfg.n_layers, capacity, cfg.n_kv_heads, cfg.d_head);

    // ---- Prefill：把整段 prompt 一次性喂进去，pos_offset=0 ----
    std::vector<int> total_ids = prompt_ids; // 预测结果序列
    int pos_offset = 0;   // 已经进过 cache 的 token 数

    Tensor pref_logits({(int)prompt_ids.size(), cfg.vocab_size});
    transformer_forward_kv(prompt_ids, w, cfg, pref_logits, cache, pos_offset);
    pos_offset += (int)prompt_ids.size();

    // 取 prompt 最后一个 token 的 logits，greedy 选下一个
    int next = argmax(pref_logits.data() + (prompt_ids.size() - 1) * cfg.vocab_size, cfg.vocab_size);
    total_ids.push_back(next);

    // ---- Decode：每次只喂一个新 token，pos_offset=i ----
    for (int i = 0; i < max_new_tokens-1; ++i){
        std::vector<int> one = {next};
        Tensor logits({1, cfg.vocab_size});
        transformer_forward_kv(one, w, cfg, logits, cache, pos_offset);
        pos_offset += 1;

        next = argmax(logits.data(), cfg.vocab_size);
        total_ids.push_back(next);
    }

    return total_ids;
}