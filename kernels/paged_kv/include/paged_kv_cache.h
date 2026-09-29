#pragma once
#include <vector>
#include "block.h"
#include "block_table.h"

namespace paged_kv {

class PagedKVCache {
    public:
        // 池总量 = n_layers * num_blocks * block_size * n_kv_heads * d_head；
        // 每层内布局 [num_blocks, block_size, n_kv_heads, d_head] 行优先。
        PagedKVCache(int nLayers, int nKvHeads, int dHead, int numBlocks, int blockSize);

        // 把逻辑位置 [start_pos, start_pos+n_tokens) 的 K/V 写进 table 指向的块。
        // 前置: start_pos+n_tokens <= table.capacity_tokens()  (写"已分配区"; 越界 NB_CHECK)
        void write(int layer, const BlockTable& table, int startPos, const float* kSrc, const float* vSrc, int nTokens);

        // 把逻辑位置 [0, num_tokens) 收集成连续缓冲。
        // 前置: num_tokens <= table.num_tokens()
        void gather(int layer, const BlockTable& table, int numTokens, float* kDst, float* vDst) const;

        // 裸块指针（Stage C / Lab 2 用）
        float* k_block(int layer, BlockId b);
        const float* k_block(int layer, BlockId b) const;
        float* v_block(int layer, BlockId b);
        const float* v_block(int layer, BlockId b) const;
        
        // 常驻 gather 缓冲 (避免每层每步 malloc + 零初始化)。
        // 大小 = max_context * n_kv_heads * d_head; 由 prepare 时 resize(不 shrink)。
        void gather_into_scratch(int layer, const BlockTable& table, int numTokens);
        const float* k_scratch() const { return k_scratch_.data(); }
        const float* v_scratch() const { return v_scratch_.data(); }
    
    private:
        // 每块的元素数 = block_size * n_kv_heads * d_head
        int block_elems() const { return blockSize_ * nKvHeads_ * dHead_; }
        
        // 单个 token 的元素数 = n_kv_heads * d_head
        int token_elems() const { return nKvHeads_ * dHead_; }

        // [n_layers][num_blocks*block_elems()]
        std::vector<std::vector<float>> kPool_, vPool_;
        std::vector<float> k_scratch_, v_scratch_;   // 常驻 gather 缓冲
        int nLayers_;
        int nKvHeads_;
        int dHead_;
        int numBlocks_;
        int blockSize_;
};

} // namespace paged_kv
