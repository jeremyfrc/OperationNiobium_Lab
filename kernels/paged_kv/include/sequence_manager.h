#pragma once
#include "block_table.h"
#include "paged_kv_cache.h"

namespace paged_kv {

    // 序列级编排: 把"分配容量 → 写 K/V → 提交逻辑长度"编成一个事务。
// 不碰 attention; 站在 BlockTable + PagedKVCache 之上。
bool append_kv(PagedKVCache& cache, BlockTable& table, int layer,
               const float* k_new, const float* v_new, int n_new);

} // namespace paged_kv