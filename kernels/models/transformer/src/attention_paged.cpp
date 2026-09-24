#include "attention_paged.h"
#include "ops.h"          // matmul, softmax_inplace
#include "rope.h"         // rope_inplace
#include "check.h"
#include <cmath>
#include <vector>

bool attention_paged_forward(const Tensor& x, const AttentionWeights& w, const TransformerConfig& cfg, Tensor& out, paged_kv::PagedKVCache& cache, paged_kv::BlockTable& table, int layer_idx, int pos_offset) {
    int seq_len = x.shape()[0];
    int d_model = cfg.d_model, n_heads = cfg.n_heads, n_kv_heads = cfg.n_kv_heads, d_head = cfg.d_head;

    NB_CHECK(d_model == n_heads * d_head, "attention: d_model must equal n_heads*d_head");
    NB_CHECK(n_heads % n_kv_heads == 0, "attention: n_heads must be divisible by n_kv_heads (GQA)");
    NB_CHECK(x.shape()[1] == d_model, "attention: input last dim must be d_model");
    NB_CHECK(w.wQ.shape()[0] == d_model && w.wQ.shape()[1] == n_heads * d_head, "attention: wQ shape [d_model, n_heads*d_head]");
    NB_CHECK(w.wK.shape()[0] == d_model && w.wK.shape()[1] == n_kv_heads * d_head, "attention: wK shape [d_model, n_kv_heads*d_head]");
    NB_CHECK(w.wV.shape()[0] == d_model && w.wV.shape()[1] == n_kv_heads * d_head, "attention: wV shape [d_model, n_kv_heads*d_head]");
    NB_CHECK(w.wO.shape()[0] == n_heads * d_head && w.wO.shape()[1] == d_model, "attention: wO shape [n_heads*d_head, d_model]");
    NB_CHECK(pos_offset >= 0, "attention_paged_forward(): offset does not allow moving backward.");

    Tensor q_linear({seq_len, n_heads * d_head});
    Tensor k_linear({seq_len, n_kv_heads * d_head});
    Tensor v_linear({seq_len, n_kv_heads * d_head});
    matmul(x, w.wQ, q_linear);
    matmul(x, w.wK, k_linear);
    matmul(x, w.wV, v_linear);

    q_linear.reshape({seq_len * n_heads, d_head});
    k_linear.reshape({seq_len * n_kv_heads, d_head});
    rope_inplace(q_linear, n_heads, d_head, pos_offset, cfg.rope_theta);
    rope_inplace(k_linear, n_kv_heads, d_head, pos_offset, cfg.rope_theta);
    q_linear.reshape({seq_len, n_heads * d_head});
    k_linear.reshape({seq_len, n_kv_heads * d_head});


    cache.write(layer_idx, table, pos_offset, k_linear.data(), v_linear.data(), seq_len);

    int total_ctx = pos_offset + seq_len;
    std::vector<float> k_buf((size_t)total_ctx * n_kv_heads * d_head);
    std::vector<float> v_buf((size_t)total_ctx * n_kv_heads * d_head);
    cache.gather(layer_idx, table, total_ctx, k_buf.data(), v_buf.data());

    // 5. attention 数学
    Tensor ctx({seq_len, n_heads * d_head});
    attention_over_kv(q_linear, k_buf.data(), v_buf.data(), ctx, cfg, seq_len, pos_offset, total_ctx);

    // 6. wO
    matmul(ctx, w.wO, out);
    return true;

}