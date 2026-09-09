#pragma once
#include "tensor.h"

// RoPE 旋转位置编码算子（In-place 原地计算）
// x: 输入并直接覆写的 Tensor，Shape 为 [seq_len, num_heads, head_dim] 或 [seq_len * num_heads, head_dim]
//  ---- 契约： x最后一维必须等于head_dim(内部会断言)。
// num_heads: 注意力头数量
// head_dim: 单个头的维度（显示传入，绝不从shape反推 -- 见check）
// pos_offset: 起始位置偏移量（Prefill 为 0，Decode 阶段为当前 KV Cache 长度）
// base: RoPE 频率基数，通常为 10000.0f 或 500000.0f
void rope_inplace(Tensor& x, int numHeads, int headDim, int posOffset = 0, float base = 10000.0f);

// Out-of-place 接口（如果需要保持兼容）
void rope(const Tensor& x, Tensor& out, int numHeads, int headDim, int posOffset = 0, float base = 10000.0f);