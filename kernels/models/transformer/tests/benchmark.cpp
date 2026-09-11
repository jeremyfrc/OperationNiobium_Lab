#include "transformer.h"
#include "test_utils.h"
#include <iostream>
#include <chrono>
#include <vector>


// 测量一次生成耗时（毫秒）
static double time_generate(bool use_kv,
                            const std::vector<int>& prompt,
                            const TransformerWeights& weights,
                            const TransformerConfig& cfg,
                            int new_tokens,
                            std::vector<int>& out_ids)
{
    auto start = std::chrono::high_resolution_clock::now();

    if (use_kv) {
        out_ids = generate_kv(prompt, weights, cfg, new_tokens);
    } else {
        out_ids = generate(prompt, weights, cfg, new_tokens);
    }

    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

int main() {
    TransformerConfig cfg = get_tiny_config();
    TransformerWeights weights = load_tiny_weights(cfg);

    std::vector<int> prompt = {1, 5, 42};
    int new_tokens = 20;

    // ---- 跑非 KV 版 ----
    std::vector<int> ids_nokv;
    double ms_nokv = time_generate(false, prompt, weights, cfg, new_tokens, ids_nokv);

    // ---- 跑 KV 版 ----
    std::vector<int> ids_kv;
    double ms_kv = time_generate(true, prompt, weights, cfg, new_tokens, ids_kv);

    // ---- 输出性能 ----
    std::cout << "===== Benchmark (new_tokens=" << new_tokens << ") =====\n";
    std::cout << "Non-KV : " << ms_nokv << " ms  ("
              << (ms_nokv / new_tokens) << " ms/token, "
              << (1000.0 * new_tokens / ms_nokv) << " tok/s)\n";
    std::cout << "KV     : " << ms_kv   << " ms  ("
              << (ms_kv / new_tokens) << " ms/token, "
              << (1000.0 * new_tokens / ms_kv) << " tok/s)\n";

    if (ms_kv > 0)
        std::cout << "Speedup (Non-KV / KV): " << (ms_nokv / ms_kv) << "x\n";

    // ---- 顺带验证两条路径结果一致 ----
    bool same = (ids_kv.size() == ids_nokv.size());
    if (same)
        for (size_t i = 0; i < ids_kv.size(); ++i)
            if (ids_kv[i] != ids_nokv[i]) { same = false; break; }

    std::cout << (same ? "✅ outputs identical\n" : "❌ outputs DIFFER (KV bug!)\n");

    return 0;
}