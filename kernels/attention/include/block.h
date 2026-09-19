#pragma once
#include <cstdint>

namespace attn {

// 物理块编号。用有符号整型，便于用 kInvalidBlock = -1 表示"无效/分配失败"。
using BlockId = int32_t;

// 分配失败 / 无效块的哨兵。allocate() 用尽时返回它（不抛不崩）。
inline constexpr BlockId kInvalidBlock = -1;

}  // namespace attn
