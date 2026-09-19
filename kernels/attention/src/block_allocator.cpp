#include "block_allocator.h"
#include "check.h"


namespace attn{

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

void BlockAllocator::free(BlockId id) {

    NB_CHECK(valid_id(id), "This id is invalid!");
    NB_CHECK(refCount_[id] == 0, "We can't free a block being referenced!");
    freeList_.push_back(id);
    assert_invariant();
    refCount_[id] = 0;
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

void BlockAllocator::assert_invariant() const {

    int active = 0;
    for (int c : refCount_) {
        if (c > 0) ++active;
    }
    NB_CHECK(num_free() + active == numBlocks_, "invariant broken.");
}

int BlockAllocator::num_allocated() const {

    return numBlocks_ - num_free();
}

} // attn
