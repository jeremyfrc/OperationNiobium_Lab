#include "block_allocator.h"
#include "check.h"


namespace paged_kv{

BlockAllocator::BlockAllocator(int numBlocks, int blockSize, ZeroRefPolicy policy): numBlocks_(numBlocks), blockSize_(blockSize), policy_(policy) {
    
    freeList_.reserve(numBlocks_);

    for (BlockId id = 0; id < numBlocks_; ++id) {
        freeList_.push_back(id);
    }
    
    refCount_.assign(numBlocks_, 0);
}

BlockId BlockAllocator::allocate() {
    if (freeList_.empty()) {
        return kInvalidBlock;
    }

    BlockId  id = freeList_.back();
    freeList_.pop_back();
    refCount_[id] = 1;
    assert_invariant();
    return id;
}

void BlockAllocator::incref(BlockId id) {
    
    NB_CHECK(valid_id(id), "This id is invalid!");
    NB_CHECK(refCount_[id] > 0, "We can't incref a free block.");
    ++refCount_[id];
    assert_invariant();
}

void BlockAllocator::decref(BlockId id) {

    NB_CHECK(valid_id(id), "This id is invalid!");
    NB_CHECK(refCount_[id] > 0, "We can't decref a free block.");
    --refCount_[id];
    if (refCount_[id] == 0) {
        switch (policy_) {
            case ZeroRefPolicy::FreeImmediately:
                freeList_.push_back(id);
                break;
            case ZeroRefPolicy::MoveToEvictable:
                NB_CHECK(false, "not implemented");
                break;
        }
    }
    assert_invariant();
}

int BlockAllocator::refcount(BlockId id) const {

    NB_CHECK(valid_id(id), "This id is invalid!");
    return refCount_[id];
}

bool BlockAllocator::check_invariant() const {
    // 1) 计数不变式 : num_free() + (#refcount > 0) == num_blocks
    int active = 0;
    for (int c: refCount_) if (c > 0) ++active;
    if (num_free() + active != numBlocks_) return false;

    // 2) free list 无重复，且不与已分配块重叠(= 全集且互斥)
    std::vector<char> seen(numBlocks_, 0);
    for (BlockId id: freeList_) {
        if (!valid_id(id)) return false;
        if (seen[id]) return false;
        if (refCount_[id] != 0) return false;
        seen[id] = 1;
    }
    for (int id = 0; id < numBlocks_; ++id) {
        if (refCount_[id] > 0) {
            if (seen[id]) return false; // 已分配块不在 free list
        } else {
            // refcount == 0的块必须在free list
            if (!seen[id]) return false;
        }
    }
    return true;
}

void BlockAllocator::assert_invariant() const {
#ifndef NDEBUG
    NB_CHECK(check_invariant(), "allocator invariant broken.");
#endif
}

int BlockAllocator::num_allocated() const {

    return numBlocks_ - num_free();
}

} // namespace paged_kv
