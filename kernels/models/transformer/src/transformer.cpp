#include "transformer.h"
#include "rmsnorm.h"
#include "attention.h"
#include "swiglu.h"
#include "ops.h"
#include <vector>
#include <cassert>

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
    // TODO: 调用 rmsnorm(x, w.rms1_weight, cfg.rms_eps, x_norm1);
    rmsnorm(x, w.rms1_weight, x_norm1, cfg.rms_eps);

    // 2. Attention
    Tensor attn_out({seq_len, d_model});
    // TODO: 调用 attention_forward(x_norm1, w.attn, cfg, attn_out);
    attention_forward(x_norm1, w.attn, cfg, attn_out);

    // 3. Residual Connection 1 (x = x + attn_out)
    // 注意：为了不破坏输入的 x（如果后面还要用），我们先拷贝一份或者直接输出到 out
    std::copy(x.data(), x.data() + x.numel(), out.data());
    // TODO: 调用 add_inplace(out, attn_out);
    add_inplace(out, attn_out);

    // 4. Pre-FFN RMSNorm
    Tensor x_norm2({seq_len, d_model});
    // TODO: 调用 rmsnorm(out, w.rms2_weight, cfg.rms_eps, x_norm2);
    rmsnorm(out, w.rms2_weight, x_norm2, cfg.rms_eps);
    // 5. Feed Forward (SwiGLU)
    Tensor ffn_out({seq_len, d_model});
    // TODO: 调用 swiglu_forward(x_norm2, w.ffn, ffn_out);
    swiglu_forward(x_norm2, w.ffn, ffn_out);

    // 6. Residual Connection 2 (out = out + ffn_out)
    // TODO: 调用 add_inplace(out, ffn_out);
    add_inplace(out, ffn_out);
}

// 整栈前向: token_ids -> Embedding -> Layers -> Final Norm -> LM Head -> Logits
void transformer_forward(const std::vector<int>& token_ids,
                         const TransformerWeights& w,
                         const TransformerConfig& cfg,
                         Tensor& logits) {
    int seq_len = token_ids.size();
    int d_model = cfg.d_model;

    // 1. Embedding Lookup
    // 将 token_ids 转换为 [seq_len, d_model] 的 Tensor
    Tensor x({seq_len, d_model});
    
    // TODO: 实现 Embedding 查表逻辑
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
    std::copy(x.data(), x.data() + x.numel(), buffer_b.data());
    Tensor* input_ptr = &buffer_a;
    Tensor* output_ptr = &buffer_b;

    for (int l = 0; l < cfg.n_layers; ++l) {
        Tensor current_layer_out({seq_len, d_model});
        // TODO: 调用 decoder_layer_forward(layer_in_out, w.layers[l], cfg, current_layer_out);
        decoder_layer_forward(*input_ptr, w.layers[l], cfg, *output_ptr);
        // 更新 layer_in_out = current_layer_out，准备下一层
        std::swap(input_ptr, output_ptr);
    }

    // 3. Final RMSNorm
    Tensor x_final_norm({seq_len, d_model});
    // TODO: 调用 rmsnorm(layer_in_out, w.final_rms_weight, cfg.rms_eps, x_final_norm);
    rmsnorm(*input_ptr, w.final_rms_weight, x_final_norm, cfg.rms_eps);

    // 4. LM Head (Linear to Logits)
    // logits 维度应该是 [seq_len, vocab_size]
    // TODO: 调用 matmul(x_final_norm, w.lm_head, logits);
    matmul(x_final_norm, w.lm_head, logits);
}

// Stage 4 才会用到的生成逻辑，现在可以留空
std::vector<int> generate(const std::vector<int>& prompt_ids,
                          const TransformerWeights& w,
                          const TransformerConfig& cfg,
                          int max_new_tokens) {
    // TODO: 实现自回归生成逻辑
    return std::vector<int>();
}