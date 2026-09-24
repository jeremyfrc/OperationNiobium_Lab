// Stage A — BlockTable 单测
// 覆盖 spec 验收判据的检查点 3/4/5/6:
//   - locate 跨块正确性
//   - ensure_capacity 失败必须全回滚(不留半成品)
//   - clone_shared 后每块 refcount==2; 原表析构后克隆表仍有效
//   - 所有表析构后 num_free()==num_blocks (零泄漏)

#include "block_allocator.h"
#include "block_table.h"

#include <cstdio>
#include <utility>

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

// ---- 检查点 3: locate 跨块 ----
// block_size=4, 8 token -> 2 块
//   locate(0)=(b0,0) locate(3)=(b0,3) locate(4)=(b1,0) locate(7)=(b1,3)
static void test_locate() {
  BlockAllocator alloc(8, 4);
  BlockTable table(&alloc);

  CHECK(table.ensure_capacity(8));
  CHECK(table.num_blocks() == 2);

  const BlockId b0 = table.blocks()[0];
  const BlockId b1 = table.blocks()[1];

  CHECK(table.locate(0) == std::make_pair(b0, 0));
  CHECK(table.locate(3) == std::make_pair(b0, 3));
  CHECK(table.locate(4) == std::make_pair(b1, 0));
  CHECK(table.locate(7) == std::make_pair(b1, 3));

  alloc.assert_invariant();
}

// ---- ensure_capacity 幂等: 再次请求更小容量不应多分配 ----
static void test_ensure_idempotent() {
  BlockAllocator alloc(8, 4);
  BlockTable table(&alloc);

  CHECK(table.ensure_capacity(8));   // 2 块
  const int freeAfterFirst = alloc.num_free();

  CHECK(table.ensure_capacity(4));   // 只要 1 块, 已够
  CHECK(table.num_blocks() == 2);
  CHECK(alloc.num_free() == freeAfterFirst);
}

// ---- 检查点 4: ensure_capacity 失败要完全回滚 ----
static void test_ensure_oom() {
  BlockAllocator alloc(8, 4);
  BlockTable table(&alloc);

  CHECK(table.ensure_capacity(4));   // 先占 1 块
  CHECK(table.num_blocks() == 1);
  const int freeBefore = alloc.num_free();

  // 请求远超 8 块的容量 -> 中途 allocate 失败
  const bool ok = table.ensure_capacity(999);
  CHECK(ok == false);
  CHECK(alloc.num_free() == freeBefore);   // 真回滚, 账目恢复
  CHECK(table.num_blocks() == 1);          // 没留半成品

  // allocator 状态未被破坏: 后续仍能正常分配
  BlockId id = alloc.allocate();
  CHECK(id != kInvalidBlock);
  alloc.decref(id);

  alloc.assert_invariant();
}

// ---- append_tokens ----
static void test_append_tokens() {
  BlockAllocator alloc(8, 4);
  BlockTable table(&alloc);
  CHECK(table.ensure_capacity(8));     // 2 块容量 = 8 token
  CHECK(table.num_tokens() == 0);

  table.append_tokens(5);
  CHECK(table.num_tokens() == 5);
  table.append_tokens(3);
  CHECK(table.num_tokens() == 8);
}

// ---- 检查点 5: clone_shared 后每块 refcount==2; 析构其一, 另一仍可用 ----
static void test_clone_shared() {
  BlockAllocator alloc(8, 4);

  // 让"克隆表"活得比"原表"久: 用内层作用域关掉 original
  BlockTable survive(&alloc);
  {
    BlockTable original(&alloc);
    CHECK(original.ensure_capacity(8));     // 占 2 块
    original.append_tokens(8);

    survive = original.clone_shared();      // 浅拷; 每块 incref -> 2
    for (BlockId id : original.blocks()) CHECK(alloc.refcount(id) == 2);
    CHECK(survive.num_tokens() == 8);
    CHECK(survive.blocks() == original.blocks());
  }                                          // original 析构 -> 每块 decref -> 2->1

  // 原表已析构, 克隆表依然能正确 locate, refcount 回到 1
  for (BlockId id : survive.blocks()) CHECK(alloc.refcount(id) == 1);
  CHECK(survive.locate(0) == std::make_pair(survive.blocks()[0], 0));
  CHECK(survive.locate(7) == std::make_pair(survive.blocks()[1], 3));

  alloc.assert_invariant();
}

// ---- 检查点 6: 所有表析构后清零 (单独作用域, 验证零泄漏) ----
static void test_no_leak() {
  BlockAllocator alloc(8, 4);
  {
    BlockTable a(&alloc);
    CHECK(a.ensure_capacity(8));
    a.append_tokens(8);
    BlockTable b = a.clone_shared();
    CHECK(alloc.num_free() == 6);         // 8 - 2
    (void)b;
  }                                       // a、b 都析构
  CHECK(alloc.num_free() == 8);           // 全部归还
  alloc.assert_invariant();
}

int main() {
  test_locate();
  test_ensure_idempotent();
  test_ensure_oom();
  test_append_tokens();
  test_clone_shared();
  test_no_leak();

  if (g_failed == 0) {
    std::printf("test_allocate_table: ALL PASSED\n");
    return 0;
  }
  std::printf("test_allocate_table: %d CHECK(s) FAILED\n", g_failed);
  return 1;
}


