// Stage B — PagedKVCache 单测
// 覆盖 spec Stage B 验收判据:
//   1. 写 17 个 token(跨 5 块) 再 gather -> 与源 bit-identical
//   2. 跨块边界正确: 第 4/第 5 个 token 分属不同块, 值不串
//   3. 两个 BlockTable 用同一 allocator 交错分配, 各自 gather 互不污染
//   4. write / gather 越界 -> NB_CHECK 抛 std::runtime_error 拦下
//   5. write 的契约是"已分配容量"(capacity_tokens), 而非"已提交长度"(num_tokens):
//      - 写"已分配但未 append"的区 -> 合法 (append_kv 依赖此行为)
//      - 超出已分配容量 -> 抛
#include "block_allocator.h"
#include "block_table.h"
#include "paged_kv_cache.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

using namespace attn;

static int g_failed = 0;
#include <sys/wait.h>
#include <unistd.h>

#define CHECK(cond)                                                       \
  do {                                                                    \
    if (!(cond)) {                                                        \
      std::fprintf(stderr, "CHECK failed: %s (%s:%d)\n", #cond,           \
                   __FILE__, __LINE__);                                   \
      ++g_failed;                                                         \
    }                                                                     \
  } while (0)

// 测试配置 (spec 指定, 小到能手算)
static constexpr int kNLayers  = 2;
static constexpr int kNKvHeads = 2;
static constexpr int kDHead    = 4;
static constexpr int kBlockSz  = 4;
static constexpr int kNumBlk   = 8;
static constexpr int kTokElems = kNKvHeads * kDHead;   // 8

// 确定性伪随机: 不用 rand, 保证可复现 / 可手算
static float genK(int layer, int pos, int e) { return 1.f + layer * 1000.f + pos * 10.f + e; }
static float genV(int layer, int pos, int e) { return 2.f + layer * 1000.f + pos * 10.f + e; }

// 把 [start, start+n) 的"预期值"填进连续缓冲 (k 或 v)
static void fill_expected(std::vector<float>& dst, bool isK, int layer, int start, int n) {
  dst.assign((size_t)n * kTokElems, 0.f);
  for (int t = 0; t < n; ++t)
    for (int e = 0; e < kTokElems; ++e)
      dst[(size_t)t * kTokElems + e] =
          isK ? genK(layer, start + t, e) : genV(layer, start + t, e);
}

// 构造一段连续源缓冲, 供 write 使用
static void make_src(std::vector<float>& src, bool isK, int layer, int start, int n) {
  fill_expected(src, isK, layer, start, n);
}

// ---- 验收1: 写 17 token(跨 5 块) -> gather -> bit-identical ----
static void test_roundtrip_17() {
  BlockAllocator alloc(kNumBlk, kBlockSz);
  BlockTable table(&alloc);
  PagedKVCache cache(kNLayers, kNKvHeads, kDHead, kNumBlk, kBlockSz);

  const int n = 17;
  CHECK(table.ensure_capacity(n));   // ceil(17/4)=5 块
  table.append_tokens(n);

  std::vector<float> kSrc, vSrc;
  make_src(kSrc, true,  0, 0, n);
  make_src(vSrc, false, 0, 0, n);

  cache.write(0, table, /*startPos=*/0, kSrc.data(), vSrc.data(), n);

  std::vector<float> kOut((size_t)n * kTokElems), vOut((size_t)n * kTokElems);
  cache.gather(0, table, n, kOut.data(), vOut.data());

  CHECK(std::memcmp(kOut.data(), kSrc.data(), kSrc.size() * sizeof(float)) == 0);
  CHECK(std::memcmp(vOut.data(), vSrc.data(), vSrc.size() * sizeof(float)) == 0);
}

// ---- 验收2: 跨块边界不串 (block_size=4; 第4/第5个 token 分属不同块) ----
static void test_cross_block_boundary() {
  BlockAllocator alloc(kNumBlk, kBlockSz);
  BlockTable table(&alloc);
  PagedKVCache cache(kNLayers, kNKvHeads, kDHead, kNumBlk, kBlockSz);

  const int n = 8;
  CHECK(table.ensure_capacity(n));   // 2 块
  table.append_tokens(n);

  std::vector<float> kSrc, vSrc;
  make_src(kSrc, true,  1, 0, n);
  make_src(vSrc, false, 1, 0, n);
  cache.write(1, table, 0, kSrc.data(), vSrc.data(), n);

  // 第 3 号和 第 4 号逻辑 token 分别落在 block 0 尾 / block 1 头
  const BlockId b0 = table.blocks()[0];
  const BlockId b1 = table.blocks()[1];
  const float* k0 = cache.k_block(1, b0);
  const float* k1 = cache.k_block(1, b1);

  for (int e = 0; e < kTokElems; ++e) {
    CHECK(k0[(size_t)3 * kTokElems + e] == genK(1, 3, e));   // 块0 第3格 = token3
    CHECK(k1[(size_t)0 * kTokElems + e] == genK(1, 4, e));   // 块1 第0格 = token4
  }
}

// ---- 验收3: 两表交错分配, 各自 gather 不污染 ----
static void test_interleaved_two_tables() {
  BlockAllocator alloc(kNumBlk, kBlockSz);
  BlockTable a(&alloc);
  BlockTable b(&alloc);
  PagedKVCache cache(kNLayers, kNKvHeads, kDHead, kNumBlk, kBlockSz);

  // 交错: A 拿 2 块 -> B 拿 2 块 -> A 再拿 2 块
  CHECK(a.ensure_capacity(2 * kBlockSz));   // A: 2 块
  CHECK(b.ensure_capacity(2 * kBlockSz));   // B: 2 块
  CHECK(a.ensure_capacity(3 * kBlockSz));   // A: 再 1 块 -> 共 3 块

  const int nA = 3 * kBlockSz;   // A 写满 3 块 = 12 token
  const int nB = 2 * kBlockSz;   // B 写满 2 块 = 8 token
  a.append_tokens(nA);
  b.append_tokens(nB);

  std::vector<float> kA, vA, kB, vB;
  make_src(kA, true,  0, 0, nA);  make_src(vA, false, 0, 0, nA);
  make_src(kB, true,  1, 0, nB);  make_src(vB, false, 1, 0, nB);

  cache.write(0, a, 0, kA.data(), vA.data(), nA);
  cache.write(0, b, 0, kB.data(), vB.data(), nB);

  std::vector<float> kAOut((size_t)nA * kTokElems), vAOut((size_t)nA * kTokElems);
  std::vector<float> kBOut((size_t)nB * kTokElems), vBOut((size_t)nB * kTokElems);
  cache.gather(0, a, nA, kAOut.data(), vAOut.data());
  cache.gather(0, b, nB, kBOut.data(), vBOut.data());

  CHECK(std::memcmp(kAOut.data(), kA.data(), kA.size() * sizeof(float)) == 0);
  CHECK(std::memcmp(vAOut.data(), vA.data(), vA.size() * sizeof(float)) == 0);
  CHECK(std::memcmp(kBOut.data(), kB.data(), kB.size() * sizeof(float)) == 0);
  CHECK(std::memcmp(vBOut.data(), vB.data(), vB.size() * sizeof(float)) == 0);

  // A 的 3 块物理号应互不相同, 且与 B 的 2 块互不相同
  CHECK(a.blocks().size() == 3);
  CHECK(b.blocks().size() == 2);
  for (size_t i = 0; i < a.blocks().size(); ++i)
    for (size_t j = i + 1; j < a.blocks().size(); ++j)
      CHECK(a.blocks()[i] != a.blocks()[j]);
}

// ---- 验收4a: write 越界 -> NB_CHECK 抛异常 ----
static void test_write_out_of_range() {
  BlockAllocator alloc(kNumBlk, kBlockSz);
  BlockTable table(&alloc);
  PagedKVCache cache(kNLayers, kNKvHeads, kDHead, kNumBlk, kBlockSz);
  table.ensure_capacity(kBlockSz);
  table.append_tokens(kBlockSz);   // num_tokens=4
  std::vector<float> kSrc((size_t)8 * kTokElems, 1.f);
  std::vector<float> vSrc((size_t)8 * kTokElems, 1.f);

  bool threw = false;
  try {
    cache.write(0, table, 0, kSrc.data(), vSrc.data(), 8);   // 8 > 4: 越界
  } catch (const std::runtime_error&) {
    threw = true;
  }
  CHECK(threw);
}

// ---- 验收4b: gather 越界 -> NB_CHECK 抛异常 ----
static void test_gather_out_of_range() {
  BlockAllocator alloc(kNumBlk, kBlockSz);
  BlockTable table(&alloc);
  PagedKVCache cache(kNLayers, kNKvHeads, kDHead, kNumBlk, kBlockSz);
  table.ensure_capacity(kBlockSz);
  table.append_tokens(kBlockSz);   // num_tokens=4
  std::vector<float> kDst((size_t)8 * kTokElems), vDst((size_t)8 * kTokElems);

  bool threw = false;
  try {
    cache.gather(0, table, 8, kDst.data(), vDst.data());     // 8 > 4: 越界
  } catch (const std::runtime_error&) {
    threw = true;
  }
  CHECK(threw);
}

// ---- 验收5a: write 到"已分配但未提交"的区 -> 合法 (append_kv 依赖此语义) ----
// 关键: 这里 startPos=4, n=4, 但 num_tokens 仍为 4 (只提交了前半)。
// 旧契约(num_tokens) 会误判越界; 新契约(capacity_tokens=8) 应放行。
static void test_write_into_allocated_uncommitted() {
  BlockAllocator alloc(kNumBlk, kBlockSz);
  BlockTable table(&alloc);
  PagedKVCache cache(kNLayers, kNKvHeads, kDHead, kNumBlk, kBlockSz);

  CHECK(table.ensure_capacity(2 * kBlockSz));   // 2 块 -> 容量 8
  table.append_tokens(kBlockSz);                // 只提交 4 -> num_tokens=4

  std::vector<float> kSrc, vSrc;
  make_src(kSrc, true,  0, 4, kBlockSz);   // token 4..7 的数据
  make_src(vSrc, false, 0, 4, kBlockSz);

  bool threw = false;
  try {
    cache.write(0, table, /*startPos=*/4, kSrc.data(), vSrc.data(), kBlockSz);  // [4,8) 已分配, 合法
  } catch (const std::runtime_error&) {
    threw = true;
  }
  CHECK(!threw);   // 不应抛

  // 提交后再 gather 全部 8 个, 前 4 个是 0(没写过), 后 4 个应是刚写的
  table.append_tokens(kBlockSz);   // num_tokens = 8
  std::vector<float> kOut((size_t)8 * kTokElems, -1.f), vOut((size_t)8 * kTokElems, -1.f);
  cache.gather(0, table, 8, kOut.data(), vOut.data());
  // 前 4 个 token 从没写过 -> 池初始化为 0
  for (int e = 0; e < 4 * kTokElems; ++e) {
    CHECK(kOut[e] == 0.f);
    CHECK(vOut[e] == 0.f);
  }
  // 后 4 个 token = 刚写进 [4,8) 的数据
  CHECK(std::memcmp(kOut.data() + 4 * kTokElems, kSrc.data(), kSrc.size() * sizeof(float)) == 0);
  CHECK(std::memcmp(vOut.data() + 4 * kTokElems, vSrc.data(), vSrc.size() * sizeof(float)) == 0);
}

// ---- 验收5b: 超出"已分配容量" -> 抛 (新契约真正拦的点) ----
static void test_write_beyond_capacity() {
  BlockAllocator alloc(kNumBlk, kBlockSz);
  BlockTable table(&alloc);
  PagedKVCache cache(kNLayers, kNKvHeads, kDHead, kNumBlk, kBlockSz);

  CHECK(table.ensure_capacity(kBlockSz));   // 容量 4
  // 不 append: num_tokens=0, capacity=4

  std::vector<float> kSrc((size_t)8 * kTokElems, 1.f);
  std::vector<float> vSrc((size_t)8 * kTokElems, 1.f);

  bool threw = false;
  try {
    cache.write(0, table, 0, kSrc.data(), vSrc.data(), 8);   // 8 > 容量 4 -> 抛
  } catch (const std::runtime_error&) {
    threw = true;
  }
  CHECK(threw);
}

int main() {
  test_roundtrip_17();
  test_cross_block_boundary();
  test_interleaved_two_tables();
  test_write_out_of_range();
  test_gather_out_of_range();
  test_write_into_allocated_uncommitted();
  test_write_beyond_capacity();

  if (g_failed == 0) {
    std::printf("test_paged_kv_cache: ALL PASSED\n");
    return 0;
  }
  std::printf("test_paged_kv_cache: %d CHECK(s) FAILED\n", g_failed);
  return 1;
}