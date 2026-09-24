#pragma once
#include "block_table.h"
#include "paged_kv_cache.h"

namespace paged_kv {

// 序列级: 保证 table 能装下 (num_tokens + n) 个 token, 并把逻辑长度推进 n。
// 只碰 table (块映射 + 长度), 不碰任何 layer 的 K/V 数据。
// 所有 layer 共享一个 table -> 此函数在整个序列前向里【只调用一次】, 且在
// 任何层的 cache.write 之前。
// 返回 false: 容量不足(块用完), table 保持调用前状态(ensure_capacity 内部已回滚)。
bool prepare_sequence(BlockTable& table, int n);

}  // namespace paged_kv