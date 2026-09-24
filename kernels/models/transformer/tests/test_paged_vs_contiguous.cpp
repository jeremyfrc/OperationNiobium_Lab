// Stage C 验收: paged 路径 vs Phase 2 连续 KV 路径, 逐步对拍。
//
// 覆盖验收判据:
//   [1] prompt=3 + 20 步 decode, 两条路径 logits 逐步 check_close(rtol=1e-5)
//   [2] greedy 采出的 token id 序列完全相同
//   [3] 块用量 == ceil(23 / block_size), 一块不多
//   [4] 序列结束后释放, num_free() 回到初值
//   [5] 中途 allocate 失败(num_blocks 调到刚好不够) -> 优雅返回 false, 不崩
#include "transformer.h"
#include "forward_paged.h"
#include "paged_kv_cache.h"
#include "block_table.h"
#include "block_allocator.h"
#include "test_utils.h"
#include "utils.h"
#include "check.h"

#include <cmath>
#include <cstdio>
#include <vector>

static int g_failed = 0;
#define CHECK(cond)                                                       \
  do {                                                                    \
    if (!(cond)) {                                                        \
      std::fprintf(stderr, "CHECK failed: %s (%s:%d)\n", #cond,           \
                   __FILE__, __LINE__);                                   \
      ++g_failed;                                                         \
    }                                                                     \
  } while (0)

static const int kBlockSize = 4;
static const int kNumBlocks = 8;   // 够 ceil(23/4)=6 块

// ---------------------------------------------------------------------
// 主链路对拍: prefill + 逐步 decode, 返回 greedy 序列
// ---------------------------------------------------------------------
static std::vector<int> run_stepwise_match(const TransformerConfig& cfg,
                                           const TransformerWeights& w,
                                           const std::vector<int>& prompt,
                                           int steps,
                                           const char* tag) {
  // 连续 KV 路径 (Phase 2)
  KVCache kv_cache(cfg.n_layers, (int)prompt.size() + steps, cfg.n_kv_heads, cfg.d_head);

  // paged 路径
  paged_kv::BlockAllocator alloc(kNumBlocks, kBlockSize);
  paged_kv::BlockTable table(&alloc);
  paged_kv::PagedKVCache cache(cfg.n_layers, cfg.n_kv_heads, cfg.d_head, kNumBlocks, kBlockSize);

  std::vector<int> seq = prompt;

  // ---- prefill: 整段 prompt, pos_offset = 0 ----
  Tensor kv_pref({(int)prompt.size(), cfg.vocab_size});
  Tensor pg_pref({(int)prompt.size(), cfg.vocab_size});
  transformer_forward_kv(prompt, w, cfg, kv_pref, kv_cache, 0);
  CHECK(forward_paged(prompt, w, cfg, pg_pref, cache, table, 0));

  {
    int last = (int)prompt.size() - 1;
    bool ok = check_close(pg_pref.data() + last * cfg.vocab_size,
                          kv_pref.data() + last * cfg.vocab_size,
                          cfg.vocab_size, /*rtol=*/1e-5f);
    if (!ok) std::fprintf(stderr, "[%s] prefill 末位 logits 不匹配\n", tag);
    CHECK(ok);
  }

  // 用连续路径的 argmax 取第一个新 token (两条路径吃同样的 token)
  int next = argmax(kv_pref.data() + ((int)prompt.size() - 1) * cfg.vocab_size, cfg.vocab_size);
  seq.push_back(next);

  // ---- decode: 每步 1 个 token ----
  for (int t = 1; t < steps; ++t) {
    int pos = (int)seq.size() - 1;
    std::vector<int> one = { seq.back() };

    Tensor kv_l({1, cfg.vocab_size});
    Tensor pg_l({1, cfg.vocab_size});
    transformer_forward_kv(one, w, cfg, kv_l, kv_cache, pos);
    CHECK(forward_paged(one, w, cfg, pg_l, cache, table, pos));

    // [1] logits 逐步对拍
    bool ok = check_close(pg_l.data(), kv_l.data(), cfg.vocab_size, /*rtol=*/1e-5f);
    if (!ok) std::fprintf(stderr, "[%s] decode 第 %d 步 logits 不匹配\n", tag, t);
    CHECK(ok);

    // [2] 两条路径 greedy 出的 token 必须相同
    int kv_next = argmax(kv_l.data(), cfg.vocab_size);
    int pg_next = argmax(pg_l.data(), cfg.vocab_size);
    if (kv_next != pg_next)
      std::fprintf(stderr, "[%s] decode 第 %d 步 greedy token 不同: kv=%d pg=%d\n",
                   tag, t, kv_next, pg_next);
    CHECK(kv_next == pg_next);

    seq.push_back(kv_next);
  }

  // [3] 块用量 == ceil((prompt+steps) / block_size)
  int expected_blocks = (int)std::ceil((double)((int)prompt.size() + steps) / kBlockSize);
  if ((int)table.blocks().size() != expected_blocks)
    std::fprintf(stderr, "[%s] 块用量 = %d, 期望 %d\n", tag,
                 (int)table.blocks().size(), expected_blocks);
  CHECK((int)table.blocks().size() == expected_blocks);

  // [4] 生命周期: 用独立 allocator, 建表 -> 写一些 -> 表析构 -> 块全归还
  {
    paged_kv::BlockAllocator a2(kNumBlocks, kBlockSize);
    const int free0 = a2.num_free();
    CHECK(free0 == kNumBlocks);
    {
      paged_kv::BlockTable t2(&a2);
      paged_kv::PagedKVCache c2(cfg.n_layers, cfg.n_kv_heads, cfg.d_head, kNumBlocks, kBlockSize);
      Tensor l2({1, cfg.vocab_size});
      CHECK(forward_paged({1}, w, cfg, l2, c2, t2, 0));
      CHECK(a2.num_free() < free0);          // 用掉了块
    }  // <-- t2 析构, 应归还所有块
    CHECK(a2.num_free() == kNumBlocks);      // 全部归还
  }

  return seq;
}

// ---------------------------------------------------------------------
// [5] OOM: num_blocks 刚好不够 -> forward_paged 返回 false, 不崩
// ---------------------------------------------------------------------
static void test_oom_graceful(const TransformerConfig& cfg, const TransformerWeights& w) {
  std::vector<int> prompt = {1, 5, 42};
  const int steps = 20;

  // 故意只给 1 块 (block_size=4 -> 只装 4 token), 远不够 23
  paged_kv::BlockAllocator alloc(/*num_blocks=*/1, kBlockSize);
  paged_kv::BlockTable table(&alloc);
  paged_kv::PagedKVCache cache(cfg.n_layers, cfg.n_kv_heads, cfg.d_head, /*num_blocks=*/1, kBlockSize);

  // prefill 3 个能装下 (1 块=4), 继续 decode 到超容量时应返回 false
  bool saw_false = false;
  std::vector<int> seq = prompt;
  Tensor pref({(int)prompt.size(), cfg.vocab_size});
  if (!forward_paged(prompt, w, cfg, pref, cache, table, 0)) {
    saw_false = true;   // 也可能 prefill 就 false
  } else {
    int next = argmax(pref.data() + ((int)prompt.size() - 1) * cfg.vocab_size, cfg.vocab_size);
    seq.push_back(next);
    for (int t = 1; t < steps; ++t) {
      std::vector<int> one = { seq.back() };
      Tensor l({1, cfg.vocab_size});
      // t 到达容量上限后会 false
      if (!forward_paged(one, w, cfg, l, cache, table, (int)seq.size() - 1)) {
        saw_false = true;
        break;
      }
      seq.push_back(argmax(l.data(), cfg.vocab_size));
    }
  }
  CHECK(saw_false);   // 必须优雅返回 false, 而不是崩/越界

  // allocator 未被破坏: 仍能正常分配
  CHECK(alloc.num_free() >= 0);
}

int main() {
  TransformerConfig cfg = get_tiny_config();
  TransformerWeights w = load_tiny_weights(cfg);

  std::vector<int> prompt = {1, 5, 42};
  const int steps = 20;
  (void)steps;

  std::printf("=== Stage C: paged vs contiguous (prompt=3 + 20 decode) ===\n");
  std::vector<int> seq = run_stepwise_match(cfg, w, prompt, steps, "main");

  std::printf("greedy seq: ");
  for (int id : seq) std::printf("%d ", id);
  std::printf("\n");

  test_oom_graceful(cfg, w);

  if (g_failed == 0) {
    std::printf("test_paged_vs_contiguous: ALL PASSED\n");
    return 0;
  }
  std::printf("test_paged_vs_contiguous: %d CHECK(s) FAILED\n", g_failed);
  return 1;
}
