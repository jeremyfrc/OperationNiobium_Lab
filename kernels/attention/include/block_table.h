#pragma once
#include <vector>
#include <utility>

#include "block.h"

namespace attn {

class BlockAllocator;  // 前向声明；生命周期由外部保证

// BlockTable: 单个序列的 逻辑 token -> 物理 block 的映射。
//
// 语义（Stage A）:
//   - 每个逻辑 block 持有一份对物理块的引用（由 allocator 的 refcount 记）。
//   - clone_shared(): 浅拷贝 blocks_ + 对每块 incref()，两个表共享物理块。
//     支持 checkpoint / 分叉：释放其中一个表不会影响另一个。
//   - 析构: 对 blocks_ 里每块 decref()，归零则由 allocator 回收。
//
// 复制语义：只允许"移动"，不允许隐式拷贝。
//   clone_shared() 是唯一复制入口，它显式 incref；若允许隐式拷贝构造
//   而不 incref，析构时会双重 decref，refcount 变负。
class BlockTable {
 public:
  BlockTable(BlockAllocator* alloc);
  ~BlockTable();

  BlockTable(const BlockTable&) = delete;
  BlockTable& operator=(const BlockTable&) = delete;
  BlockTable(BlockTable&&) noexcept = default;
  BlockTable& operator=(BlockTable&&) noexcept = default;

  // 保证能容纳 num_tokens 个 token；差几块补几块。
  // 中途 allocate() 返回 kInvalidBlock 时，把本次已分配的块全部 decref 回滚，
  // 返回 false（不留半成品），allocator 状态回到调用前。
  bool ensure_capacity(int num_tokens);

  // token_pos -> 物理块 + 块内偏移。
  //   block  = blocks_[token_pos / alloc_->block_size()]
  //   offset = token_pos % alloc_->block_size()
  std::pair<BlockId, int> locate(int token_pos) const;

  // 逻辑长度 += n（调用前必须已 ensure_capacity）。只推进计数，不动物理块。
  void append_tokens(int n);

  // 浅拷贝 blocks_，对每块 incref()，num_tokens_ 一起拷贝（不动物理数据）。
  BlockTable clone_shared() const;

  int num_tokens() const { return numTokens_; }
  int block_size() const;
  int num_blocks() const { return static_cast<int>(blocks_.size()); }
  const std::vector<BlockId>& blocks() const { return blocks_; }

 private:
  BlockAllocator* alloc_;
  int numTokens_ = 0;
  std::vector<BlockId> blocks_;  // logical block idx -> physical BlockId
};

}  // namespace attn
