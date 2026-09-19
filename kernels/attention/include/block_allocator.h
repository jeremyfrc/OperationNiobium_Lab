#pragma once
#include <list>
#include <unordered_map>
#include <vector>

#include "block.h"
#include "check.h"

namespace attn {

// decref 归零时的处置策略（Stage A-1 只实现第一种）。
enum class ZeroRefPolicy {
  FreeImmediately,  // 归零 → 直接回 free_list_
  MoveToEvictable,  // 归零 → 进 evictable_（Stage A-2，暂未实现）
};

// BlockAllocator: 用 free-list 管理物理 block id 的分配 / 引用计数 / 回收。
//
// 策略（稳定可复现，单测依赖）:
//   LIFO（栈语义）：allocate() 从尾部 pop，free() 压回尾部。
//   于是 allocate -> free -> allocate 会复用同一个 block_id。
//
// 引用计数约定:
//   allocate() 即视为持有一份引用：弹出后立即 refcount_[id] = 1。
//   调用方用 incref()/decref() 增减；decref() 归零时按 policy_ 处置。
//   这样"已分配" == refcount_[id] > 0，不变式随时成立。
class BlockAllocator {
 public:
  explicit BlockAllocator(int numBlocks, int blockSize,
                          ZeroRefPolicy policy = ZeroRefPolicy::FreeImmediately);

  // 从 free_list_ 尾部弹出一个物理 id，并将其 refcount_ 置 1。
  // 用尽时返回 kInvalidBlock（不抛不崩）。
  BlockId allocate();

  // 把 block_id 归还回 free_list_（断言：id 合法）。
  void free(BlockId id);

  int num_free() const { return static_cast<int>(freeList_.size()); }
  int num_blocks() const { return numBlocks_; }
  int block_size() const { return blockSize_; }

  // 引用计数增减。
  void incref(BlockId id);
  void decref(BlockId id);  // 归零 -> 按 policy_ 处置
  int refcount(BlockId id) const;

  ZeroRefPolicy zero_ref_policy() const { return policy_; }

  // 记账不变式：num_free() + (refcount_ > 0 的块数) == num_blocks_。
  void assert_invariant() const;

 private:
  int numBlocks_;
  int blockSize_;              // 块内 token 数
  std::vector<int> freeList_;  // 空闲物理 id；尾部 = 栈顶
  std::vector<int> refCount_;   // 每个物理 id 的引用计数

  // ===== Stage A-2 占位：布局现在就焊死，接 evict 时只改分支、不动这里 =====
  ZeroRefPolicy policy_;
  std::list<BlockId> evictable_;
  std::unordered_map<BlockId, std::list<BlockId>::iterator> evictIter_;

  int num_allocated() const;  // = numBlocks_ - numFree_

  bool valid_id(BlockId id) const { return id >= 0 && id < numBlocks_; }
};

}  // namespace attn

