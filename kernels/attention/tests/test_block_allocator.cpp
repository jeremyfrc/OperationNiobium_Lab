// Stage A — BlockAllocator 单测
// 覆盖 spec 验收判据的检查点 1/2:
//   - 分配满 num_blocks 后 num_free()==0; 再 allocate() 返回 kInvalidBlock(不崩不抛)
//   - decref 到 0 -> 块回 free list, num_free() 恢复; LIFO 复用同一 id
//   - 引用计数 / 不变式

#include "block_allocator.h"

#include <cstdio>
#include <vector>

using namespace attn;

static int g_failed = 0;
#define CHECK(cond)                                                       \
  do {                                                                    \
    if (!(cond)) {                                                        \
      std::fprintf(stderr, "CHECK failed: %s (%s:%d)\n", #cond,           \
                   __FILE__, __LINE__);                                   \
      ++g_failed;                                                         \
    }                                                                     \
  } while (0)

// ---- 检查点 1: 分配满 8 块 ----
static void test_alloc_full() {
  BlockAllocator alloc(8, 4);
  CHECK(alloc.num_free() == 8);
  CHECK(alloc.num_blocks() == 8);
  CHECK(alloc.block_size() == 4);

  // LIFO: 依次弹出 7,6,5,4,3,2,1,0
  std::vector<BlockId> got;
  for (int i = 0; i < 8; ++i) got.push_back(alloc.allocate());

  for (int i = 0; i < 8; ++i) CHECK(got[i] == 7 - i);
  CHECK(alloc.num_free() == 0);

  // 池空: 返回 kInvalidBlock, 不崩不抛
  CHECK(alloc.allocate() == kInvalidBlock);
  CHECK(alloc.num_free() == 0);
}

// ---- 检查点 2: decref 归零 -> 回 freeList, 复用同一 id ----
static void test_decref_reuse() {
  BlockAllocator alloc(8, 4);

  BlockId a = alloc.allocate();        // 7
  CHECK(alloc.refcount(a) == 1);
  CHECK(alloc.num_free() == 7);

  alloc.decref(a);                     // 归零 -> 回 freeList
  CHECK(alloc.num_free() == 8);
  CHECK(alloc.refcount(a) == 0);

  BlockId b = alloc.allocate();        // LIFO: 又是 7
  CHECK(b == a);
}

// ---- incref / decref 多引用计数 ----
static void test_refcount() {
  BlockAllocator alloc(8, 4);

  BlockId id = alloc.allocate();       // refcount=1
  CHECK(alloc.refcount(id) == 1);

  alloc.incref(id);                    // 2 (clone_shared 会这么干)
  CHECK(alloc.refcount(id) == 2);

  alloc.decref(id);                    // 1, 不回 freeList
  CHECK(alloc.refcount(id) == 1);
  CHECK(alloc.num_free() == 7);

  alloc.decref(id);                    // 0 -> 回 freeList
  CHECK(alloc.refcount(id) == 0);
  CHECK(alloc.num_free() == 8);
}

// ---- 不变式: 各种操作后都自洽 ----
static void test_invariant() {
  BlockAllocator alloc(8, 4);
  alloc.assert_invariant();

  std::vector<BlockId> ids;
  for (int i = 0; i < 5; ++i) ids.push_back(alloc.allocate());
  alloc.assert_invariant();

  alloc.incref(ids[0]);
  alloc.assert_invariant();

  alloc.decref(ids[0]);               // 2 -> 1, 仍被引用
  alloc.assert_invariant();

  alloc.decref(ids[0]);               // 1 -> 0, 回 freeList
  alloc.assert_invariant();

  // 回收其余(注意 ids[0] 已回收, 不能再 decref)
  for (int i = 1; i < (int)ids.size(); ++i) alloc.decref(ids[i]);
  alloc.assert_invariant();

  CHECK(alloc.num_free() == 8);
}

int main() {
  test_alloc_full();
  test_decref_reuse();
  test_refcount();
  test_invariant();

  if (g_failed == 0) {
    std::printf("test_block_allocator: ALL PASSED\n");
    return 0;
  }
  std::printf("test_block_allocator: %d CHECK(s) FAILED\n", g_failed);
  return 1;
}
