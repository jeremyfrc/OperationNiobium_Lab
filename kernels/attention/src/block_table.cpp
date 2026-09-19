#include "block_table.h"

#include "block_allocator.h"
#include "check.h"

#include <algorithm>

namespace attn {

BlockTable::BlockTable(BlockAllocator* alloc) : alloc_(alloc) {}

BlockTable::~BlockTable() {
    for (BlockId id: blocks_) alloc_->decref(id);
}

bool BlockTable::ensure_capacity(int numTokens) {

    const int blockSize = alloc_->block_size();
    const int need = (numTokens + blockSize - 1) / blockSize;
    const int have = static_cast<int>(blocks_.size());
    if (need <= have) return true;

    const int toAdd = need - have;
    std::vector<BlockId> newly;
    for (int i = 0; i < toAdd; ++i){
        BlockId id = alloc_->allocate();
        if (id == kInvalidBlock) {
            for (BlockId b: newly) alloc_->decref(b);
            blocks_.resize(have);
            return false;
        }
        newly.push_back(id);
        blocks_.push_back(id);
    }
    return true;
}

std::pair<BlockId, int> BlockTable::locate(int tokenPos) const {

    const int blockSize = alloc_->block_size();
    NB_CHECK(tokenPos >= 0 && tokenPos / blockSize < (int)blocks_.size(), "Token index out of boundary!");
    const int blockIdx = tokenPos / blockSize;
    return { blocks_[blockIdx], tokenPos % blockSize };
}

void BlockTable::append_tokens(int n) {

    NB_CHECK(n >= 0, "We can't append negative number of tokens!");
    NB_CHECK((numTokens_ + n) <= (int)blocks_.size() * alloc_->block_size(), "The requested number of tokens make it over capacity!");
    numTokens_ += n;
}

BlockTable BlockTable::clone_shared() const {

    BlockTable clone(alloc_);
    clone.blocks_ = blocks_;
    clone.numTokens_ = numTokens_;
    for (BlockId id : clone.blocks_) alloc_->incref(id);
    return clone;
}

} // namespace attn
