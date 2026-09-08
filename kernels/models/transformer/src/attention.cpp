#include "tensor.h"
#include "attention.h"
#include "config.h"
#include "ops.h"
#include "rope.h"
#include <cmath>
#include <cassert>
#include <iostream>

void attention_forward(const Tensor& x, const AttentionWeights& w, const TransformerConfig& cfg, Tensor& out) {
    int seq_len = x.shape()[0];
    int d_model = cfg.d_model;
    int n_heads = cfg.n_heads;
    int n_kv_heads = cfg.n_kv_heads;
    int d_head = cfg.d_head;
    int kv_group_size = n_heads / n_kv_heads;

    assert(d_model == n_heads * d_head && "d_model must equal n_heads * d_head");

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
    rope_inplace(q_linear, n_heads, 0, cfg.rope_theta);
    rope_inplace(k_linear, n_kv_heads, 0, cfg.rope_theta);
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
                    float q_val = q_linear.data()[i * (n_heads * d_head) + h * d_head + d];
                    float k_val = k_linear.data()[j* (n_kv_heads * d_head) + kv_h * d_head + d];
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
                    float v_val = v_linear.data()[j * (n_kv_heads * d_head) + kv_h * d_head + d];
                    sum += s_val * v_val;
                }
                c_linear.data()[i * (n_heads * d_head) + h * d_head + d] = sum;
            }
        }
    }
  
    // 6. Final Projection (Wo)
    matmul(c_linear, w.wO, out);
}