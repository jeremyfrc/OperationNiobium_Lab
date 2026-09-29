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

static std::vector<int> generate_kv_into(const std::vector<int>& prompt,
                                         const TransformerWeights& w,
                                         const TransformerConfig& cfg,
                                         int new_tokens,
                                         KVCache& cache) {
  std::vector<int> ids = prompt;
  int pos = 0;
  Tensor pref({(int)prompt.size(), cfg.vocab_size});
  transformer_forward_kv(prompt, w, cfg, pref, cache, pos);
  pos += (int)prompt.size();
  int next = argmax(pref.data() + (prompt.size() - 1) * cfg.vocab_size, cfg.vocab_size);
  ids.push_back(next);
  for (int i = 0; i < new_tokens - 1; ++i) {
    std::vector<int> one = {next};
    Tensor l({1, cfg.vocab_size});
    transformer_forward_kv(one, w, cfg, l, cache, pos);
    pos += 1;
    next = argmax(l.data(), cfg.vocab_size);
    ids.push_back(next);
  }
  return ids;
}


int main() {
    TransformerConfig cfg = get_tiny_config();
    TransformerWeights weights = load_tiny_weights(cfg);

    std::vector<int> prompt = {1, 5, 42};
    const int kRuns = 100;
    const int kNumBlocks = 128, kBlockSize = 4;   // 512 token 容量

    // 返回 min / median / IQR (min 最能反映"纯计算", 抖动多为 malloc/调度叠加)
    auto stats = [](std::vector<double> v, double& mn, double& med, double& iqr) {
        std::sort(v.begin(), v.end());
        mn = v.front();
        med = v[v.size() / 2];
        iqr = v[(v.size() * 3) / 4] - v[v.size() / 4];
    };

    std::cout << "===== Length sweep (prompt=3, env outside timer, runs="
              << kRuns << ") =====\n";
    std::cout << "new_tokens |  KV: min/med/IQR(us)        |  Paged: min/med/IQR(us)     | d-min(us) | d-med(us)\n";
    std::cout << "-----------+------------------------------+------------------------------+-----------+----------\n";

    for (int nt : {20, 60, 120, 250}) {
        const int capacity = (int)prompt.size() + nt;
        std::vector<int> ids_kv, ids_pg;
        std::vector<double> t_kv, t_pg;

        // ---- KV: KVCache 建在计时区外, 每次复用 ----
        {
            KVCache cache(cfg.n_layers, capacity, cfg.n_kv_heads, cfg.d_head);
            generate_kv_into(prompt, weights, cfg, nt, cache);   // warmup
            for (int r = 0; r < kRuns; ++r) {
                // 复用同一个 cache: 每次重新 prefill 会覆盖 [0, ...), 数值一致
                auto s = std::chrono::high_resolution_clock::now();
                ids_kv = generate_kv_into(prompt, weights, cfg, nt, cache);
                auto e = std::chrono::high_resolution_clock::now();
                t_kv.push_back(std::chrono::duration<double, std::milli>(e - s).count());
            }
        }

        // ---- Paged: allocator/table/cache 建在计时区外, 每次复用 ----
        {
            paged_kv::BlockAllocator alloc(kNumBlocks, kBlockSize);
            paged_kv::BlockTable table(&alloc);
            paged_kv::PagedKVCache cache(cfg.n_layers, cfg.n_kv_heads, cfg.d_head, kNumBlocks, kBlockSize);
            generate_paged(prompt, weights, cfg, nt, cache, table);   // warmup
            for (int r = 0; r < kRuns; ++r) {
                // 复用: 每次重新 prefill。注意 table 会累积 -> 每次需清空
                table = paged_kv::BlockTable(&alloc);
                auto s = std::chrono::high_resolution_clock::now();
                ids_pg = generate_paged(prompt, weights, cfg, nt, cache, table);
                auto e = std::chrono::high_resolution_clock::now();
                t_pg.push_back(std::chrono::duration<double, std::milli>(e - s).count());
            }
        }
        double mn_kv, med_kv, iqr_kv, mn_pg, med_pg, iqr_pg;
        stats(t_kv, mn_kv, med_kv, iqr_kv);
        stats(t_pg, mn_pg, med_pg, iqr_pg);

        // us 单位 (ms * 1000)
        auto us = [](double ms) { return ms * 1000.0; };
        std::cout << nt
                  << "         | " << us(mn_kv) << "/" << us(med_kv) << "/" << us(iqr_kv)
                  << " | " << us(mn_pg) << "/" << us(med_pg) << "/" << us(iqr_pg)
                  << " | " << us(mn_pg - mn_kv)
                  << " | " << us(med_pg - med_kv) << "\n";

        bool same = (ids_kv.size() == ids_pg.size());
        if (same)
            for (size_t i = 0; i < ids_kv.size(); ++i)
                if (ids_kv[i] != ids_pg[i]) { same = false; break; }
        std::cout << (same ? "  ✅ identical\n" : "  ❌ DIFFER!\n");
    }
    return 0;
}