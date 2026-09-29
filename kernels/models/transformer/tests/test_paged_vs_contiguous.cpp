// Stage D 验收: paged 路径 vs Phase 2 连续 KV 路径 + 边界加固。
//
// 覆盖验收判据 (spec §5):
//   [1] 数值对拍: prompt=3 + 20 步 decode, paged vs 连续 KV 逐步 logits
//       check_close(rtol=1e-5); greedy token id 序列完全相同
//   [2] 碎片化: 3 条序列 (5/11/3) 交错 allocate -> 数据不串 + 账目对 + 归还回初值
//   [3] 越界/OOM: num_blocks 刚好不够 -> ensure_capacity 返回 false, 不越界,
//       状态不破坏 (OOM 后仍能正常 allocate/free) —— 验记账不变式
//   [4] refcount 生命周期: clone_shared -> 析构原表 -> 克隆可读 ->
//       析构克隆 -> 块归还
//   [5] 主判据沿用组 1; benchmark 在 benchmark.cpp (release)
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
#include <cstring>
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
// [1] 数值对拍: prefill + 20 步 decode, 返回 greedy 序列
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
  for (int t = 1; t <= steps; ++t) {
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

  // 块用量 == ceil((prompt+steps) / block_size)
  int expected_blocks = (int)std::ceil((double)((int)prompt.size() + steps) / kBlockSize);
  if ((int)table.blocks().size() != expected_blocks)
    std::fprintf(stderr, "[%s] 块用量 = %d, 期望 %d\n", tag,
                 (int)table.blocks().size(), expected_blocks);
  CHECK((int)table.blocks().size() == expected_blocks);

  return seq;
}

// ---------------------------------------------------------------------
// [4] 生命周期: 建表 -> 写 -> 表析构 -> 块全归还
// ---------------------------------------------------------------------
static void test_lifecycle(const TransformerConfig& cfg, const TransformerWeights& w) {
  paged_kv::BlockAllocator alloc(kNumBlocks, kBlockSize);
  const int free0 = alloc.num_free();
  CHECK(free0 == kNumBlocks);
  {
    paged_kv::BlockTable t2(&alloc);
    paged_kv::PagedKVCache c2(cfg.n_layers, cfg.n_kv_heads, cfg.d_head, kNumBlocks, kBlockSize);
    Tensor l2({1, cfg.vocab_size});
    CHECK(forward_paged({1}, w, cfg, l2, c2, t2, 0));
    CHECK(alloc.num_free() < free0);            // 用掉了块
    CHECK(alloc.check_invariant());
  }  // <-- t2 析构, 应归还所有块
  CHECK(alloc.num_free() == kNumBlocks);        // 全部归还
  CHECK(alloc.check_invariant());
}

// ---------------------------------------------------------------------
// [2] 碎片化: 3 条序列 (5/11/3 token) 交错增长, 各写各的数据
//     块在池中交错分配 -> 各序列块不连续; 断言数据不串 + 账目对 + 归还
// ---------------------------------------------------------------------
static const int kFragBlockSize = 4;
static const int kFragNumBlocks = 12;   // 5->2 + 11->3 + 3->1 = 6 块, 余量充足

// 按 token 位置生成可区分数据 (tok 索引 -> 值)
static float frag_val(int tag, int pos, int e) {
  return (float)(tag * 100000 + pos * 100 + e);
}

static void test_fragmentation() {
  const int layers = 1, kvHeads = 2, dHead = 4, tokElems = kvHeads * dHead;
  paged_kv::BlockAllocator alloc(kFragNumBlocks, kFragBlockSize);
  paged_kv::PagedKVCache cache(layers, kvHeads, dHead, kFragNumBlocks, kFragBlockSize);

  paged_kv::BlockTable tA(&alloc);   // 5 token
  paged_kv::BlockTable tB(&alloc);   // 11 token
  paged_kv::BlockTable tC(&alloc);   // 3 token

  // ---- 交错增长 (制造碎片) ----
  CHECK(tA.ensure_capacity(1));      // A: 1 token -> 1 块
  CHECK(tB.ensure_capacity(4));      // B: 4 token -> 1 块
  CHECK(tC.ensure_capacity(3));      // C: 3 token -> 1 块
  CHECK(tA.ensure_capacity(5));      // A 长到 5 -> 再 1 块 (共 2)
  CHECK(tB.ensure_capacity(11));     // B 长到 11 -> 再 2 块 (共 3)
  CHECK(alloc.check_invariant());

  // 逻辑长度推进 (原 prepare_sequence 语义; 这里直接操纵 table, 不走 forward)
  tA.append_tokens(5);
  tB.append_tokens(11);
  tC.append_tokens(3);

  CHECK((int)tA.blocks().size() == 2);   // ceil(5/4)
  CHECK((int)tB.blocks().size() == 3);   // ceil(11/4)
  CHECK((int)tC.blocks().size() == 1);   // ceil(3/4)
  CHECK(alloc.num_free() == kFragNumBlocks - 6);
  CHECK(alloc.check_invariant());

  // ---- 各写各的数据 ----
  const int tags[3] = {1, 2, 3};
  paged_kv::BlockTable* tabs[3] = {&tA, &tB, &tC};
  const int lens[3] = {5, 11, 3};
  for (int s = 0; s < 3; ++s) {
    std::vector<float> k((size_t)lens[s] * tokElems), v((size_t)lens[s] * tokElems);
    for (int p = 0; p < lens[s]; ++p)
      for (int e = 0; e < tokElems; ++e) {
        k[(size_t)p * tokElems + e] = frag_val(tags[s], p, e);
        v[(size_t)p * tokElems + e] = frag_val(tags[s], p, e) + 0.5f;
      }
    cache.write(0, *tabs[s], 0, k.data(), v.data(), lens[s]);
  }

  // ---- gather 回读, 验证不串 ----
  for (int s = 0; s < 3; ++s) {
    std::vector<float> kOut((size_t)lens[s] * tokElems), vOut((size_t)lens[s] * tokElems);
    cache.gather(0, *tabs[s], lens[s], kOut.data(), vOut.data());
    for (int p = 0; p < lens[s]; ++p)
      for (int e = 0; e < tokElems; ++e) {
        CHECK(kOut[(size_t)p * tokElems + e] == frag_val(tags[s], p, e));
        CHECK(vOut[(size_t)p * tokElems + e] == frag_val(tags[s], p, e) + 0.5f);
      }
  }

  // ---- 析构 A, 验证 B/C 不受影响, 且账目对 ----
  tA = paged_kv::BlockTable(&alloc);   // move-assign: 释放 A 的旧块
  CHECK(alloc.num_free() == kFragNumBlocks - 4);   // 归还 A 的 2 块
  CHECK(alloc.check_invariant());
  for (int s = 1; s < 3; ++s) {
    std::vector<float> kOut((size_t)lens[s] * tokElems);
    std::vector<float> vOut((size_t)lens[s] * tokElems);
    cache.gather(0, *tabs[s], lens[s], kOut.data(), vOut.data());
    for (int p = 0; p < lens[s]; ++p)
      for (int e = 0; e < tokElems; ++e)
        CHECK(kOut[(size_t)p * tokElems + e] == frag_val(tags[s], p, e));
  }

  // ---- 全部释放后回到初值 ----
  tA = paged_kv::BlockTable(&alloc);
  tB = paged_kv::BlockTable(&alloc);
  tC = paged_kv::BlockTable(&alloc);
  CHECK(alloc.num_free() == kFragNumBlocks);
  CHECK(alloc.check_invariant());
}

// ---------------------------------------------------------------------
// [3] 越界/OOM: num_blocks 刚好不够 -> false, 不越界, 状态不破坏
// ---------------------------------------------------------------------
static void test_oom_graceful(const TransformerConfig& cfg, const TransformerWeights& w) {
  std::vector<int> prompt = {1, 5, 42};
  const int steps = 20;

  // 池子: 5 块 * 4 = 20 token 容量; 需要 23 -> 第 6 块时失败 (刚好不够)
  const int oomBlocks = 5;
  paged_kv::BlockAllocator alloc(oomBlocks, kBlockSize);
  paged_kv::BlockTable table(&alloc);
  paged_kv::PagedKVCache cache(cfg.n_layers, cfg.n_kv_heads, cfg.d_head, oomBlocks, kBlockSize);

  // prefill 3 个能装下; decode 到超 20 时 forward_paged 应返回 false
  bool saw_false = false;
  std::vector<int> seq = prompt;
  Tensor pref({(int)prompt.size(), cfg.vocab_size});
  if (!forward_paged(prompt, w, cfg, pref, cache, table, 0)) {
    saw_false = true;
  } else {
    int next = argmax(pref.data() + ((int)prompt.size() - 1) * cfg.vocab_size, cfg.vocab_size);
    seq.push_back(next);
    for (int t = 1; t <= steps; ++t) {
      std::vector<int> one = { seq.back() };
      Tensor l({1, cfg.vocab_size});
      if (!forward_paged(one, w, cfg, l, cache, table, (int)seq.size() - 1)) {
        saw_false = true;
        break;
      }
      seq.push_back(argmax(l.data(), cfg.vocab_size));
    }
  }
  CHECK(saw_false);                       // 必须优雅返回 false, 而不是崩/越界
  CHECK(alloc.check_invariant());         // OOM 后记账不变式成立

  // spec: OOM 后状态必须"干净" —— 再走一次正常 allocate/free 验证
  // (池已满 -> 再要 1 块应失败 false, 而不是崩; 之后归还仍正常)
  {
    paged_kv::BlockTable t3(&alloc);
    bool got = t3.ensure_capacity(kBlockSize);   // 想要 1 块
    CHECK(!got);                                 // 池已满 -> 拿不到
    CHECK(alloc.check_invariant());
  }                                              // t3 析构 -> 归还(若有)
  CHECK(alloc.check_invariant());               // 归还后仍干净

  // 原表未受影响: 仍持有 5 块
  CHECK((int)table.blocks().size() == oomBlocks);
}

// ---------------------------------------------------------------------
// [5] refcount 生命周期: clone_shared, 两份表共享物理块
// ---------------------------------------------------------------------
static void test_clone_shared(const TransformerConfig& cfg, const TransformerWeights& w) {
  const int n = 3;
  paged_kv::BlockAllocator alloc(kNumBlocks, kBlockSize);
  paged_kv::PagedKVCache cache(cfg.n_layers, cfg.n_kv_heads, cfg.d_head, kNumBlocks, kBlockSize);

  // 原表在内层 scope 建 + 写, 结束时析构; 把克隆表带出来。
  paged_kv::BlockTable clone = [&]() {
    paged_kv::BlockTable orig(&alloc);
    Tensor logits({3, cfg.vocab_size});
    CHECK(forward_paged({1, 5, 42}, w, cfg, logits, cache, orig, 0));
    CHECK(orig.num_tokens() == n);
    CHECK((int)orig.blocks().size() == 1);                     // ceil(3/4)=1
    CHECK(alloc.num_free() == kNumBlocks - 1);

    paged_kv::BlockTable c = orig.clone_shared();              // 共享 -> refcount 2
    CHECK(c.num_tokens() == n);
    CHECK(alloc.refcount(orig.blocks()[0]) == 2);
    return c;                                                  // 移出 c
  }();  // orig 析构: decref, clone 仍持有 -> 块不回收

  // 原表已析构, 克隆独占块, refcount 应为 1
  CHECK(alloc.refcount(clone.blocks()[0]) == 1);
  CHECK(alloc.num_free() == kNumBlocks - 1);                   // 块未回收

  // 克隆表仍可读: gather 不越界
  const int tokElems = cfg.n_kv_heads * cfg.d_head;
  std::vector<float> kOut((size_t)n * tokElems), vOut((size_t)n * tokElems);
  cache.gather(0, clone, n, kOut.data(), vOut.data());         // 不崩即通过

  // 析构克隆表 -> 块归还
  clone = paged_kv::BlockTable(&alloc);
  CHECK(alloc.num_free() == kNumBlocks);
  CHECK(alloc.check_invariant());
}

// ---------------------------------------------------------------------
// [6] 故意喂错误的 pos_offset -> NB_CHECK 抛异常
// ---------------------------------------------------------------------
static void test_bad_pos_offset(const TransformerConfig& cfg, const TransformerWeights& w) {
  paged_kv::BlockAllocator alloc(kNumBlocks, kBlockSize);
  paged_kv::BlockTable table(&alloc);
  paged_kv::PagedKVCache cache(cfg.n_layers, cfg.n_kv_heads, cfg.d_head, kNumBlocks, kBlockSize);

  Tensor pref({3, cfg.vocab_size});
  CHECK(forward_paged({1, 5, 42}, w, cfg, pref, cache, table, 0));
  CHECK(table.num_tokens() == 3);

  Tensor bad({1, cfg.vocab_size});
  bool threw = false;
  try {
    forward_paged({7}, w, cfg, bad, cache, table, /*pos_offset=*/2);   // 错!
  } catch (const std::exception& e) {
    threw = true;
    std::printf("[bad pos_offset] caught: %s\n", e.what());
  }
  CHECK(threw);
}

int main() {
  TransformerConfig cfg = get_tiny_config();
  TransformerWeights w = load_tiny_weights(cfg);

  std::vector<int> prompt = {1, 5, 42};
  const int steps = 20;

  std::printf("=== Stage D: paged vs contiguous (prompt=3 + %d decode) ===\n", steps);
  std::vector<int> seq = run_stepwise_match(cfg, w, prompt, steps, "main");

  std::printf("greedy seq: ");
  for (int id : seq) std::printf("%d ", id);
  std::printf("\n");

  test_lifecycle(cfg, w);       // [4]
  test_fragmentation();         // [2]
  test_oom_graceful(cfg, w);    // [3]
  test_clone_shared(cfg, w);    // [5]
  test_bad_pos_offset(cfg, w);  // [6]

  if (g_failed == 0) {
    std::printf("test_paged_vs_contiguous: ALL PASSED\n");
    return 0;
  }
  std::printf("test_paged_vs_contiguous: %d CHECK(s) FAILED\n", g_failed);
  return 1;
}
