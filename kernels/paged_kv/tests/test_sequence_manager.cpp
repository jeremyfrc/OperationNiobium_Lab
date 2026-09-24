// Stage C — prepare_sequence 单测 (序列级: 只备容量 + 推进长度)
// 数据写入由各层 cache.write 负责, 这里一并验证 write+gather 往返。
// 覆盖:
//   1. 单次 prefill n=8 -> num_tokens==8, gather == 源 (bit-identical)
//   2. 增量 decode: 连续 prepare_sequence(1) + write x5 -> num_tokens 递增, gather 正确
//   3. 跨块 n=17(5 块) -> gather bit-identical
//   4. OOM: 容量不够 -> 返回 false, num_tokens 不变, allocator 未损坏
//   5. n == 0 -> 返回 true, num_tokens 不变
//   6. 失败后 table 完全不变 (blocks 数量不变, num_free 不变)
//   7. 多层共享同一 table: prepare_sequence 只调一次, 每层各写各的, 长度不翻倍
#include "block_allocator.h"
#include "block_table.h"
#include "paged_kv_cache.h"
#include "sequence_manager.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

using namespace paged_kv;

static int g_failed = 0;
#define CHECK(cond)                                                       \
  do {                                                                    \
    if (!(cond)) {                                                        \
      std::fprintf(stderr, "CHECK failed: %s (%s:%d)\n", #cond,           \
                   __FILE__, __LINE__);                                   \
      ++g_failed;                                                         \
    }                                                                     \
  } while (0)

// 测试配置 (spec 指定)
static constexpr int kNLayers  = 2;
static constexpr int kNKvHeads = 2;
static constexpr int kDHead    = 4;
static constexpr int kBlockSz  = 4;
static constexpr int kNumBlk   = 8;
static constexpr int kTokElems = kNKvHeads * kDHead;   // 8

// 确定性伪随机, 与 test_paged_kv_cache 一致
static float genK(int layer, int pos, int e) { return 1.f + layer * 1000.f + pos * 10.f + e; }
static float genV(int layer, int pos, int e) { return 2.f + layer * 1000.f + pos * 10.f + e; }

// 填 [start, start+n) 的预期值
static void make_src(std::vector<float>& dst, bool isK, int layer, int start, int n) {
  dst.assign((size_t)n * kTokElems, 0.f);
  for (int t = 0; t < n; ++t)
    for (int e = 0; e < kTokElems; ++e)
      dst[(size_t)t * kTokElems + e] =
          isK ? genK(layer, start + t, e) : genV(layer, start + t, e);
}

// ---- 1: 单次 prefill n=8 ----
static void test_prefill() {
  BlockAllocator alloc(kNumBlk, kBlockSz);
  BlockTable table(&alloc);
  PagedKVCache cache(kNLayers, kNKvHeads, kDHead, kNumBlk, kBlockSz);

  const int n = 8;
  std::vector<float> kSrc, vSrc;
  make_src(kSrc, true,  0, 0, n);
  make_src(vSrc, false, 0, 0, n);

  CHECK(prepare_sequence(table, n));            // 序列级: 备容量 + 提交长度
  CHECK(table.num_tokens() == n);
  cache.write(/*layer=*/0, table, /*start=*/0, kSrc.data(), vSrc.data(), n);   // 层级: 写数据

  std::vector<float> kOut((size_t)n * kTokElems), vOut((size_t)n * kTokElems);
  cache.gather(0, table, n, kOut.data(), vOut.data());
  CHECK(std::memcmp(kOut.data(), kSrc.data(), kSrc.size() * sizeof(float)) == 0);
  CHECK(std::memcmp(vOut.data(), vSrc.data(), vSrc.size() * sizeof(float)) == 0);
}

// ---- 2: 增量 decode, 每次 1 个 ----
static void test_incremental_decode() {
  BlockAllocator alloc(kNumBlk, kBlockSz);
  BlockTable table(&alloc);
  PagedKVCache cache(kNLayers, kNKvHeads, kDHead, kNumBlk, kBlockSz);

  const int n = 5;
  for (int t = 0; t < n; ++t) {
    std::vector<float> k1, v1;
    make_src(k1, true,  0, t, 1);   // 第 t 个 token
    make_src(v1, false, 0, t, 1);
    CHECK(prepare_sequence(table, 1));
    CHECK(table.num_tokens() == t + 1);
    cache.write(0, table, /*start=*/t, k1.data(), v1.data(), 1);
  }

  // gather 全部 5 个, 应等于 genK/ genV 的 [0,5)
  std::vector<float> kExp, vExp;
  make_src(kExp, true,  0, 0, n);
  make_src(vExp, false, 0, 0, n);
  std::vector<float> kOut((size_t)n * kTokElems), vOut((size_t)n * kTokElems);
  cache.gather(0, table, n, kOut.data(), vOut.data());
  CHECK(std::memcmp(kOut.data(), kExp.data(), kExp.size() * sizeof(float)) == 0);
  CHECK(std::memcmp(vOut.data(), vExp.data(), vExp.size() * sizeof(float)) == 0);
}

// ---- 3: 跨块 n=17 (block_size=4 -> 5 块) ----
static void test_cross_block() {
  BlockAllocator alloc(kNumBlk, kBlockSz);
  BlockTable table(&alloc);
  PagedKVCache cache(kNLayers, kNKvHeads, kDHead, kNumBlk, kBlockSz);

  const int n = 17;
  std::vector<float> kSrc, vSrc;
  make_src(kSrc, true,  1, 0, n);
  make_src(vSrc, false, 1, 0, n);

  CHECK(prepare_sequence(table, n));
  CHECK(table.num_tokens() == n);
  CHECK((int)table.blocks().size() == 5);   // ceil(17/4)
  cache.write(/*layer=*/1, table, /*start=*/0, kSrc.data(), vSrc.data(), n);

  std::vector<float> kOut((size_t)n * kTokElems), vOut((size_t)n * kTokElems);
  cache.gather(1, table, n, kOut.data(), vOut.data());
  CHECK(std::memcmp(kOut.data(), kSrc.data(), kSrc.size() * sizeof(float)) == 0);
  CHECK(std::memcmp(vOut.data(), vSrc.data(), vSrc.size() * sizeof(float)) == 0);
}

// ---- 4: OOM (容量不够) -> false, num_tokens 不变, allocator 未损坏 ----
static void test_oom() {
  // 整个池只有 2 块 = 8 token; 想提交 9 个 -> 必失败
  BlockAllocator alloc(/*numBlocks=*/2, kBlockSz);
  BlockTable table(&alloc);
  PagedKVCache cache(kNLayers, kNKvHeads, kDHead, /*numBlocks=*/2, kBlockSz);

  // 先成功 4 个
  std::vector<float> k4, v4;
  make_src(k4, true, 0, 0, 4);
  make_src(v4, false, 0, 0, 4);
  CHECK(prepare_sequence(table, 4));
  cache.write(0, table, 0, k4.data(), v4.data(), 4);
  CHECK(table.num_tokens() == 4);

  const int blocks_before = (int)table.blocks().size();
  const int free_before   = alloc.num_free();

  // 再想提交 5 个 -> 需要到 9 > 容量 8 -> false
  bool ok = prepare_sequence(table, 5);
  CHECK(ok == false);

  // table 完全不变
  CHECK(table.num_tokens() == 4);
  CHECK((int)table.blocks().size() == blocks_before);
  CHECK(alloc.num_free() == free_before);

  // allocator 未损坏: 仍然能分配 (再提交 4 个到末尾 -> 正好用满)
  std::vector<float> k4b, v4b;
  make_src(k4b, true, 0, 4, 4);
  make_src(v4b, false, 0, 4, 4);
  CHECK(prepare_sequence(table, 4));
  cache.write(0, table, 4, k4b.data(), v4b.data(), 4);
  CHECK(table.num_tokens() == 8);
  CHECK(alloc.num_free() == 0);   // 2 块用满
}

// ---- 5: n == 0 -> true, 什么都不变 ----
static void test_zero() {
  BlockAllocator alloc(kNumBlk, kBlockSz);
  BlockTable table(&alloc);
  PagedKVCache cache(kNLayers, kNKvHeads, kDHead, kNumBlk, kBlockSz);
  (void)cache;

  CHECK(prepare_sequence(table, 0));
  CHECK(table.num_tokens() == 0);
  CHECK((int)table.blocks().size() == 0);   // 没分配块
  CHECK(alloc.num_free() == kNumBlk);
}

// ---- 7: 多层共享同一 table: prepare 一次, 各层各写, 长度不翻倍 ----
static void test_multi_layer_shared_table() {
  BlockAllocator alloc(kNumBlk, kBlockSz);
  BlockTable table(&alloc);
  PagedKVCache cache(kNLayers, kNKvHeads, kDHead, kNumBlk, kBlockSz);

  const int n = 4;
  // 序列级: 只提交一次 (这正是多层场景的正确用法)
  CHECK(prepare_sequence(table, n));
  CHECK(table.num_tokens() == n);   // 不因层数翻倍

  // 每层各写各的 (用不同的 layer 值, 数据分离)
  std::vector<float> k, v;
  for (int l = 0; l < kNLayers; ++l) {
    make_src(k, true,  l, 0, n);
    make_src(v, false, l, 0, n);
    cache.write(l, table, 0, k.data(), v.data(), n);
  }

  // 每层 gather 应回各自的源 (互不影响)
  for (int l = 0; l < kNLayers; ++l) {
    make_src(k, true,  l, 0, n);
    make_src(v, false, l, 0, n);
    std::vector<float> kOut((size_t)n * kTokElems), vOut((size_t)n * kTokElems);
    cache.gather(l, table, n, kOut.data(), vOut.data());
    CHECK(std::memcmp(kOut.data(), k.data(), k.size() * sizeof(float)) == 0);
    CHECK(std::memcmp(vOut.data(), v.data(), v.size() * sizeof(float)) == 0);
  }

  CHECK((int)table.blocks().size() == 1);   // ceil(4/4) == 1, 与层数无关
}

int main() {
  test_prefill();
  test_incremental_decode();
  test_cross_block();
  test_oom();
  test_zero();
  test_multi_layer_shared_table();

  if (g_failed == 0) {
    std::printf("test_sequence_manager: ALL PASSED\n");
    return 0;
  }
  std::printf("test_sequence_manager: %d CHECK(s) FAILED\n", g_failed);
  return 1;
}
