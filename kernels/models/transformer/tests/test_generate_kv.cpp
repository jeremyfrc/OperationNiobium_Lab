#include "transformer.h"
#include "test_utils.h"
#include "kv_cache.h"
#include "config.h"
#include "utils.h"
#include <iostream>
#include <vector>

// ---------------------------------------------------------------
// KV-Cache 版生成测试
// 核心验证：KV 版 generate 与普通(非 KV) generate 在同权重、同 prompt
//           下的 greedy 输出必须逐位一致。
// ---------------------------------------------------------------
// 逐步对照 KV vs 非 KV 的 logits。
// 返回 true 表示每一步都通过 check_close(rtol=1e-5)。
static bool compare_stepwise_logits(
    const std::vector<int>& prompt,
    const TransformerWeights& weights,
    const TransformerConfig& cfg,
    int steps)
{
    // ---- KV 路径：prefill 整段 prompt，pos_offset = 0 ----
    KVCache cache(cfg.n_layers, (int)prompt.size() + steps, cfg.n_kv_heads, cfg.d_head);

    std::vector<int> seq = prompt;
    Tensor pref_logits({(int)seq.size(), cfg.vocab_size});
    transformer_forward_kv(seq, weights, cfg, pref_logits, cache, 0);

    // prefill 末位也要对照
    {
        Tensor nokv({(int)seq.size(), cfg.vocab_size});
        transformer_forward(seq, weights, cfg, nokv);
        float* nokv_last = nokv.data() + (seq.size() - 1) * cfg.vocab_size;
        float* kv_last   = pref_logits.data() + (seq.size() - 1) * cfg.vocab_size;
        if (!check_close(kv_last, nokv_last, cfg.vocab_size, /*rtol=*/1e-5f)) {
            std::cerr << "[stepwise] prefill 末位 logits 不匹配\n";
            return false;
        }
    }

    // 先用 prefill 末位取第一个新 token
    int next = argmax(pref_logits.data() + (seq.size() - 1) * cfg.vocab_size, cfg.vocab_size);
    seq.push_back(next);

    // ---- 逐步 decode 对照 ----
    for (int t = 1; t < steps; ++t) {
        int pos = (int)seq.size() - 1;   // 当前 token 的绝对位置（将被喂入）
        std::vector<int> one = { seq.back() };

        // KV：只喂一个 token
        Tensor kv_logits({1, cfg.vocab_size});
        transformer_forward_kv(one, weights, cfg, kv_logits, cache, pos);

        // 非 KV：整段重算，取最后一行
        Tensor nokv({(int)seq.size(), cfg.vocab_size});
        transformer_forward(seq, weights, cfg, nokv);
        float* nokv_last = nokv.data() + (seq.size() - 1) * cfg.vocab_size;

        if (!check_close(kv_logits.data(), nokv_last, cfg.vocab_size, /*rtol=*/1e-5f)) {
            std::cerr << "[stepwise] decode 第 " << t << " 步 logits 不匹配\n";
            return false;
        }

        next = argmax(kv_logits.data(), cfg.vocab_size);
        seq.push_back(next);
    }

    return true;
}

int main() {
    // 1. 配置和权重
    TransformerConfig cfg = get_tiny_config();
    TransformerWeights weights = load_tiny_weights(cfg);   // 默认目录 tests/ref/data/full_

    // 2. Prompt
    std::vector<int> prompt = {1, 5, 42};
    int new_tokens = 5;   // 生成的新 token 数（KV 版内部会逐 token decode）

    std::cout << "=== KV-Cache Generate Test ===" << std::endl;
    std::cout << "Prompt: ";
    for (int id : prompt) std::cout << id << " ";
    std::cout << std::endl;

    // 3. KV 版生成
    std::vector<int> kv_result = generate_kv(prompt, weights, cfg, new_tokens);

    std::cout << "KV   generated: ";
    for (int id : kv_result) std::cout << id << " ";
    std::cout << std::endl;

    // 4. 普通(非 KV)版生成，作为对照
    std::vector<int> ref_result = generate(prompt, weights, cfg, new_tokens);

    std::cout << "NonKV generated: ";
    for (int id : ref_result) std::cout << id << " ";
    std::cout << std::endl;

    // 5. 逐位比对
    bool same = (kv_result.size() == ref_result.size());
    if (same) {
        for (size_t i = 0; i < kv_result.size(); ++i) {
            if (kv_result[i] != ref_result[i]) { same = false; break; }
        }
    }

    // ---- 加固：逐 logits 对照（rtol=1e-5，比 token id 严格）----
    bool logits_ok = compare_stepwise_logits(prompt, weights, cfg, new_tokens);

    if (logits_ok) {
        std::cout << "✅ KV logits match non-KV logits (rtol=1e-5)" << std::endl;
    } else {
        std::cout << "❌ KV logits MISMATCH non-KV logits" << std::endl;
    }

    if (same && logits_ok) {
        std::cout << "✅ KV generate matches non-KV generate (tokens + logits)!" << std::endl;
        return 0;
    } else {
        std::cout << "❌ KV generate MISMATCH!" << std::endl;
        return 1;
    }
}