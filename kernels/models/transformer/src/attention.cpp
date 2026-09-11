#include "tensor.h"
#include "attention.h"
#include "config.h"
#include "ops.h"
#include "rope.h"
#include "check.h"
#include <cmath>
#include <iostream>

// 激活张量layout的单一真相源：row-major[seq, heads, head_dim] 摊平。
// 所有对q/k/b/c的手写下标都走这里，改Layout只改这一处，杜绝n_heads/n_kv_heads写反。
static inline int head_flat(int seq, int head, int d, int n_heads, int head_dim) {
    return seq * (n_heads * head_dim) + head * head_dim + d;
}

void attention_forward(const Tensor& x, const AttentionWeights& w, const TransformerConfig& cfg, Tensor& out) {
    int seq_len = x.shape()[0];
    int d_model = cfg.d_model;
    int n_heads = cfg.n_heads;
    int n_kv_heads = cfg.n_kv_heads;
    int d_head = cfg.d_head;
    int kv_group_size = n_heads / n_kv_heads;

    // P3. 入口按config 钉死形状，传错tensor 当场炸，不让他漂到输出
    NB_CHECK(d_model == n_heads * d_head, "attention: d_model must equal n_heads*d_head");
    NB_CHECK(n_heads % n_kv_heads == 0, "attention: n_heads must be divisible by n_kv_heads (GQA)");
    NB_CHECK(x.shape()[1] == d_model, "attention: input last dim must be d_model");
    NB_CHECK(w.wQ.shape()[0] == d_model && w.wQ.shape()[1] == n_heads * d_head, "attention: wQ shape [d_model, n_heads*d_head]");
    NB_CHECK(w.wK.shape()[0] == d_model && w.wK.shape()[1] == n_kv_heads * d_head, "attention: wK shape [d_model, n_kv_heads*d_head]");
    NB_CHECK(w.wV.shape()[0] == d_model && w.wV.shape()[1] == n_kv_heads * d_head, "attention: wV shape [d_model, n_kv_heads*d_head]");
    NB_CHECK(w.wO.shape()[0] == n_heads * d_head && w.wO.shape()[1] == d_model, "attention: wO shape [n_heads*d_head, d_model]");

    // 1. Q/K/V 投影 (Linear Projections)
    // matmul 期望 2D 输出，所以我们先构造 2D Tensor
    Tensor q_linear({seq_len, n_heads * d_head});
    Tensor k_linear({seq_len, n_kv_heads * d_head});
    Tensor v_linear({seq_len, n_kv_heads * d_head});

    matmul(x, w.wQ, q_linear);
    matmul(x, w.wK, k_linear);
    matmul(x, w.wV, v_linear);

    // 2. 施加 RoPE (旋转位置编码)
    // rope_inplace的契约：输入最后一维必须是head_dim
    // 投影输出是[seq_len, heads*d_head]，最后一维是heads*d_head
    // 直接传会让rope把整行当成是一个head_dim = heads*d_head的头来转
    // 位置/频率全错，先reshape成[seq*heads, d_head]，rope后再还原。
    q_linear.reshape({seq_len * n_heads, d_head});
    k_linear.reshape({seq_len * n_kv_heads, d_head});
    rope_inplace(q_linear, n_heads, d_head, 0, cfg.rope_theta);
    rope_inplace(k_linear, n_kv_heads, d_head, 0, cfg.rope_theta);
    q_linear.reshape({seq_len, n_heads * d_head});
    k_linear.reshape({seq_len, n_kv_heads * d_head});

    // 3. 计算 Attention Score 并应用 Causal Mask
    // scores 维度: [n_heads, seq_len, seq_len]
    Tensor scores({n_heads, seq_len, seq_len});
    float scale = 1.0f / std::sqrt(static_cast<float>(d_head));

    for (int h = 0; h < n_heads; ++h) {
        int kv_h = h / kv_group_size;
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < seq_len; ++j) {
                if (j > i) {
                    scores.at({h, i, j}) = -1e9f;
                    continue;
                }

                //计算 Q[i, h, :]点乘 K[j, kv_h, :]
                float sum = 0.0f;
                for (int d = 0; d < d_head; ++d){
                    float q_val = q_linear.data()[head_flat(i, h, d, n_heads, d_head)];
                    float k_val = k_linear.data()[head_flat(j, kv_h, d, n_kv_heads, d_head)];
                    sum += q_val * k_val;
                }

                scores.at({h, i, j}) = sum * scale;
            }
        }
    }
    
    // 4. Softmax (作用在最后一维 seq_len)
    softmax_inplace(scores);

    // 5. 计算 Attention Context (Score * V)
    // 先计算到 2D 线性空间，方便最后做 Wo 投影
    Tensor c_linear({seq_len, n_heads * d_head});
    for (int h = 0; h < n_heads; ++h){
        int kv_h = h / kv_group_size;
        for (int i =  0; i < seq_len; ++i) {
            for (int d = 0; d < d_head; ++d){
                float sum = 0.0f;
                for (int j = 0; j <= i; ++j){
                    float s_val = scores.at({h, i, j});
                    float v_val = v_linear.data()[head_flat(j, kv_h, d, n_kv_heads, d_head)];
                    sum += s_val * v_val;
                }
                c_linear.data()[head_flat(i, h, d, n_heads, d_head)] = sum;
            }
        }
    }
  
    // 6. Final Projection (Wo)
    matmul(c_linear, w.wO, out);
}


void attention_forward_kv(const Tensor& x, const AttentionWeights& w, const TransformerConfig& cfg, Tensor& out, KVCache& kv_cache, int layer_idx, int pos_offset) {
    int seq_len = x.shape()[0];
    int d_model = cfg.d_model;
    int n_heads = cfg.n_heads;
    int n_kv_heads = cfg.n_kv_heads;
    int d_head = cfg.d_head;
    int kv_group_size = n_heads / n_kv_heads;

    // P3. 入口按config 钉死形状，传错tensor 当场炸，不让他漂到输出
    NB_CHECK(d_model == n_heads * d_head, "attention: d_model must equal n_heads*d_head");
    NB_CHECK(n_heads % n_kv_heads == 0, "attention: n_heads must be divisible by n_kv_heads (GQA)");
    NB_CHECK(x.shape()[1] == d_model, "attention: input last dim must be d_model");
    NB_CHECK(w.wQ.shape()[0] == d_model && w.wQ.shape()[1] == n_heads * d_head, "attention: wQ shape [d_model, n_heads*d_head]");
    NB_CHECK(w.wK.shape()[0] == d_model && w.wK.shape()[1] == n_kv_heads * d_head, "attention: wK shape [d_model, n_kv_heads*d_head]");
    NB_CHECK(w.wV.shape()[0] == d_model && w.wV.shape()[1] == n_kv_heads * d_head, "attention: wV shape [d_model, n_kv_heads*d_head]");
    NB_CHECK(w.wO.shape()[0] == n_heads * d_head && w.wO.shape()[1] == d_model, "attention: wO shape [n_heads*d_head, d_model]");


    // 1. 投影计算 q, k, v (x @ w)
    Tensor q_linear({seq_len, n_heads * d_head});
    Tensor k_linear({seq_len, n_kv_heads * d_head});
    Tensor v_linear({seq_len, n_kv_heads * d_head});
    matmul(x, w.wQ, q_linear);
    matmul(x, w.wK, k_linear);
    matmul(x, w.wV, v_linear);

    // 2. RoPE: 循环 seq_len 次，每次 apply_rope(q_i, k_i, pos_offset + i)
    q_linear.reshape({seq_len * n_heads, d_head});
    rope_inplace(q_linear, n_heads, d_head, pos_offset, cfg.rope_theta);
    q_linear.reshape({seq_len, n_heads * d_head});

    k_linear.reshape({seq_len * n_kv_heads, d_head});
    rope_inplace(k_linear, n_kv_heads, d_head, pos_offset, cfg.rope_theta);
    k_linear.reshape({seq_len, n_kv_heads * d_head});

    // 3. 缓存管理:
    // 将计算出的 k, v 写入 cache.k_caches[layer_idx] 和 cache.v_caches[layer_idx]
    // 目标地址: data + (pos_offset + i) * n_kv_heads * d_head
    int kv_stride = n_kv_heads * d_head;
    float* kc = kv_cache.k_caches[layer_idx].data();
    float* vc = kv_cache.v_caches[layer_idx].data();

    for (int i = 0; i < seq_len; ++i) {
        int abs = pos_offset + i;
        std::copy(k_linear.data() + i * kv_stride, k_linear.data() + (i + 1) * kv_stride, kc + abs * kv_stride);
        std::copy(v_linear.data() + i * kv_stride, v_linear.data() + (i + 1) * kv_stride, vc + abs * kv_stride);
    }

    // 4. 这段新的输入一共会和 [0, pos_offset+seq_len) 的历史 token 做注意力
    int total_ctx = pos_offset + seq_len;

    Tensor scores({n_heads, seq_len, total_ctx});
    float scale = 1.0f / std::sqrt((float)d_head);

    for (int h = 0; h < n_heads; ++h){
        int kv_h = h / kv_group_size;
        for (int i = 0; i < seq_len; ++i){
            int abs_i = pos_offset + i;
            for (int j = 0; j < total_ctx; ++j) {
                if (j > abs_i) { scores.at({h, i, j}) = -1e9f; continue;}
                float sum = 0;
                for (int d = 0; d < d_head; ++d) {
                    float qv = q_linear.data()[head_flat(i, h, d, n_heads, d_head)];
                    float kv_ = kc[j * kv_stride + kv_h * d_head + d];
                    sum += qv * kv_;
                }
                scores.at({h, i, j}) = sum * scale;
            }
        }
    }

    // 5. softmax 沿最后一维 (total_ctx)
    softmax_inplace(scores);

    // 6. context = scores @ V，V 也从 cache 读
    Tensor ctx({seq_len, n_heads * d_head});
    for (int h = 0; h < n_heads; ++h) {
        int kv_h = h / kv_group_size;
        for (int i = 0; i < seq_len; ++i) {
            int abs_i = pos_offset + i;
            for (int d = 0; d < d_head; ++d) {
                float sum = 0;
                for (int j = 0; j <= abs_i; ++j) {
                    sum += scores.at({h, i, j}) * vc[j * kv_stride + kv_h * d_head + d];
                }
                ctx.data()[i * (n_heads * d_head) + h * d_head + d] = sum;
            }
        }
    }

    // 7. 若 seq_len>1（prefill）要 reshape 后 matmul；decode 时直接 matmul
    matmul(ctx, w.wO, out);
}