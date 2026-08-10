#pragma once
#include "tensor.h"

//对x(形状[seq_len, n_heads, d_head])原地施加旋转位置编码。
// position_offset:decode阶段用于对齐kv-cache 之后的绝对位置用的上。
void apply_rope(Tensor& x, int position_offset, float theta);