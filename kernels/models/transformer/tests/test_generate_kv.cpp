#include "transformer.h"
#include "test_utils.h"
#include "kv_cache.h"
#include "config.h"
#include <iostream>
#include <vector>

// ---------------------------------------------------------------
// KV-Cache 版生成测试
// 核心验证：KV 版 generate 与普通(非 KV) generate 在同权重、同 prompt
//           下的 greedy 输出必须逐位一致。
// ---------------------------------------------------------------

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

    if (same) {
        std::cout << "✅ KV generate matches non-KV generate!" << std::endl;
        return 0;
    } else {
        std::cout << "❌ KV generate MISMATCH with non-KV generate!" << std::endl;
        return 1;
    }
}