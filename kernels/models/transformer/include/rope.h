#pragma once
#include "tensor.h"

// RoPE 旋转位置编码算子（In-place 原地计算）
// x: 输入并直接覆写的 Tensor，Shape 为 [seq_len, num_heads, head_dim] 或 [seq_len * num_heads, head_dim]
// num_heads: 注意力头数量
// pos_offset: 起始位置偏移量（Prefill 为 0，Decode 阶段为当前 KV Cache 长度）
// base: RoPE 频率基数，通常为 10000.0f 或 500000.0f
void rope_inplace(Tensor& x, int numHeads, int posOffset = 0, float base = 10000.0f);

// Out-of-place 接口（如果需要保持兼容）
void rope(const Tensor& x, Tensor& out, int numHeads, int posOffset = 0, float base = 10000.0f);