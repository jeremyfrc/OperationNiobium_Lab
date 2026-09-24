#include "paged_kv_cache.h"
#include "check.h"
#include <cstring>
#include <algorithm>


namespace {

template<class Fn>
void for_each_span(const paged_kv::BlockTable& table, int blockSize, int startPos, int endPos, Fn f){
    int pos = startPos;
    while (pos < endPos) {
      auto [blk, off] = table.locate(pos);
      const int span = std::min(blockSize - off, endPos - pos);
      f(blk, off, span, pos);
      pos += span;
    }
}
} // namespace

namespace paged_kv {

PagedKVCache::PagedKVCache(int nLayers, int nKvHeads, int dHead, int numBlocks, int blockSize) : nLayers_(nLayers), nKvHeads_(nKvHeads), dHead_(dHead), numBlocks_(numBlocks), blockSize_(blockSize) {
    kPool_.resize(nLayers_);
    vPool_.resize(nLayers_);

    // blockSize_*nKvHeads_*dHead_有可能会超INT_MAX
    const size_t layerSize = (size_t)numBlocks_ * block_elems();
    
    for (int layer = 0; layer < nLayers_; ++layer){
      kPool_[layer].assign(layerSize, 0.f);
      vPool_[layer].assign(layerSize, 0.f);
    }
}

const float* PagedKVCache::k_block(int layer, BlockId b) const {
    NB_CHECK(layer >= 0 && layer < nLayers_, "k_block() with layer out of bound! ");
    NB_CHECK(b >= 0 && b < numBlocks_, "k_block() with BlockId out of bound! ");
    return kPool_[layer].data() + (size_t)b * block_elems();
}

float* PagedKVCache::k_block(int layer, BlockId b){
    NB_CHECK(layer >= 0 && layer < nLayers_, "k_block() with layer out of bound! ");
    NB_CHECK(b >= 0 && b < numBlocks_, "k_block() with BlockId out of bound! ");
    return kPool_[layer].data() + (size_t)b * block_elems();
}

const float* PagedKVCache::v_block(int layer, BlockId b) const {
    NB_CHECK(layer >= 0 && layer < nLayers_, "v_block() with layer out of bound! ");
    NB_CHECK(b >= 0 && b < numBlocks_, "v_block() with BlockId out of bound! ");
    return vPool_[layer].data() + (size_t)b * block_elems();
}

float* PagedKVCache::v_block(int layer, BlockId b) {
    NB_CHECK(layer >= 0 && layer < nLayers_, "v_block() with layer out of bound! ");
    NB_CHECK(b >= 0 && b < numBlocks_, "v_block() with BlockId out of bound! ");
    return vPool_[layer].data() + (size_t)b * block_elems();
}

void PagedKVCache::write(int layer, const BlockTable& table, int startPos, const float* kSrc, const float* vSrc, int nTokens) {
    
    NB_CHECK(startPos >= 0 && nTokens >= 0 && startPos + nTokens <= table.capacity_tokens(), "write(): starting / n_tokens beyong allocated capacity!");
    NB_CHECK(layer >= 0 && layer < nLayers_, "write(): layer out of range!");

    const int tokElems = token_elems();
    const int end = startPos + nTokens;

    for_each_span(table, blockSize_, startPos, end, [&](BlockId blk, int off, int span, int pos) {
      float* kDst = k_block(layer, blk) + (size_t)off * tokElems;
      const float* kNewSrc = kSrc + (size_t)(pos - startPos) * tokElems;
      std::memcpy(kDst, kNewSrc, (size_t)span * tokElems * sizeof(float));
      float* vDst = v_block(layer, blk) + (size_t)off * tokElems;
      const float* vNewSrc = vSrc + (size_t)(pos - startPos) * tokElems;
      std::memcpy(vDst, vNewSrc, (size_t)span * tokElems * sizeof(float));
    });
}

// ④ gather：把 [0, num_tokens) 逐段从块拷回连续缓冲
void PagedKVCache::gather(int layer, const BlockTable& table, int numTokens, float* kDst, float* vDst) const {

    NB_CHECK(layer >= 0 && layer < nLayers_, "gather(): layer out of boundary!");
    NB_CHECK(numTokens >= 0 && numTokens <= table.num_tokens(), "gather(): numTokens out of range!");

    const int tokElems = token_elems();

    for_each_span(table, blockSize_, 0, numTokens, [&](BlockId blk, int off, int span, int pos) {
      const float* kSrc = k_block(layer, blk) + (size_t)off * tokElems;
      std::memcpy(kDst + (size_t)pos * tokElems, kSrc, (size_t)span * tokElems * sizeof(float));
      const float* vSrc = v_block(layer, blk) + (size_t)off * tokElems;
      std::memcpy(vDst + (size_t)pos * tokElems, vSrc, (size_t)span * tokElems * sizeof(float));
    });
}

} // namespace paged_kv