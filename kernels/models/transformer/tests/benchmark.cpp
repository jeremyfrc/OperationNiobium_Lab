#include "transformer.h"
#include "forward_paged.h"       // forward_paged
#include "paged_kv_cache.h"      // PagedKVCache
#include "block_table.h"         // BlockTable
#include "block_allocator.h"     // BlockAllocator
#include "test_utils.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>


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

static std::vector<int> generate_paged(const std::vector<int>& prompt,
                                       const TransformerWeights& w,
                                       const TransformerConfig& cfg,
                                       int new_tokens,
                                       paged_kv::PagedKVCache& cache,
                                       paged_kv::BlockTable& table) {
  std::vector<int> ids = prompt;
  // prefill
  Tensor logits({(int)prompt.size(), cfg.vocab_size});
  forward_paged(prompt, w, cfg, logits, cache, table, 0);
  int last = (int)prompt.size() - 1;
  ids.push_back(argmax(logits.data() + last * cfg.vocab_size, cfg.vocab_size));
  // decode
  for (int t = 1; t < new_tokens; ++t) {
    std::vector<int> one = { ids.back() };
    Tensor l({1, cfg.vocab_size});
    if (!forward_paged(one, w, cfg, l, cache, table, (int)ids.size() - 1)) break;
    ids.push_back(argmax(l.data(), cfg.vocab_size));
  }
  return ids;
}

int main() {
    TransformerConfig cfg = get_tiny_config();
    TransformerWeights weights = load_tiny_weights(cfg);

    std::vector<int> prompt = {1, 5, 42};
    int new_tokens = 20;
    const int kRuns = 20;

    // ---- Non-KV: warmup + median (与非 KV 公平) ----
    std::vector<int> ids_nokv;
    generate(prompt, weights, cfg, new_tokens);            // warmup
    std::vector<double> t_nokv;
    for (int r = 0; r < kRuns; ++r)
        t_nokv.push_back(time_generate(false, prompt, weights, cfg, new_tokens, ids_nokv));
    std::sort(t_nokv.begin(), t_nokv.end());
    double ms_nokv = t_nokv[t_nokv.size() / 2];

    // ---- KV: warmup + median ----
    generate_kv(prompt, weights, cfg, new_tokens);         // warmup
    std::vector<double> t_kv;
    std::vector<int> ids_kv;
    for (int r = 0; r < kRuns; ++r)
        t_kv.push_back(time_generate(true, prompt, weights, cfg, new_tokens, ids_kv));
    std::sort(t_kv.begin(), t_kv.end());
    double ms_kv = t_kv[t_kv.size() / 2];

    // ---- Paged: 每次重建 allocator/table/cache, warmup + median ----
    auto run_paged_once = [&](std::vector<int>& out) -> double {
        paged_kv::BlockAllocator alloc(/*num_blocks=*/64, /*block_size=*/4);
        paged_kv::BlockTable table(&alloc);
        paged_kv::PagedKVCache cache(cfg.n_layers, cfg.n_kv_heads, cfg.d_head, 64, 4);
        auto start = std::chrono::high_resolution_clock::now();
        out = generate_paged(prompt, weights, cfg, new_tokens, cache, table);
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    };
    std::vector<int> ids_paged;
    run_paged_once(ids_paged);                              // warmup
    std::vector<double> t_paged;
    for (int r = 0; r < kRuns; ++r)
        t_paged.push_back(run_paged_once(ids_paged));
    std::sort(t_paged.begin(), t_paged.end());
    double ms_paged = t_paged[t_paged.size() / 2];

    // ---- 输出 ----
    auto row = [&](const char* name, double ms) {
        std::cout << name << " : " << ms << " ms  ("
                  << (ms / new_tokens) << " ms/token, "
                  << (1000.0 * new_tokens / ms) << " tok/s)\n";
    };
    std::cout << "===== Benchmark (new_tokens=" << new_tokens << ") =====\n";
    row("Non-KV", ms_nokv);
    row("KV    ", ms_kv);
    row("Paged ", ms_paged);

    std::cout << "Speedup (Non-KV / KV)    : " << (ms_nokv / ms_kv)    << "x\n";
    std::cout << "Speedup (Non-KV / Paged) : " << (ms_nokv / ms_paged) << "x\n";

    // ---- 三路输出一致性 ----
    auto same = [](const std::vector<int>& a, const std::vector<int>& b) {
        if (a.size() != b.size()) return false;
        for (size_t i = 0; i < a.size(); ++i) if (a[i] != b[i]) return false;
        return true;
    };
    bool ok = same(ids_nokv, ids_kv) && same(ids_kv, ids_paged);
    std::cout << (ok ? "✅ outputs identical (Non-KV == KV == Paged)\n"
                     : "❌ outputs DIFFER!\n");

    return 0;
}