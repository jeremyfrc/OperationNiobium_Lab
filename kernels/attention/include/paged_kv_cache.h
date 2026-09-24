#pragma once
#include <vector>
#include "block.h"
#include "block_table.h"

namespace attn {

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
    
    private:
        // 每块的元素数 = block_size * n_kv_heads * d_head
        int block_elems() const { return blockSize_ * nKvHeads_ * dHead_; }
        
        // 单个 token 的元素数 = n_kv_heads * d_head
        int token_elems() const { return nKvHeads_ * dHead_; }

        // [n_layers][num_blocks*block_elems()]
        std::vector<std::vector<float>> kPool_, vPool_;

        int nLayers_;
        int nKvHeads_;
        int dHead_;
        int numBlocks_;
        int blockSize_;
};

} // namespace attn
