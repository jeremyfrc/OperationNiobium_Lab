// RoPE 旋转位置编码算子（In-place 原地计算）
// x: 输入并直接覆写的 Tensor，Shape 为 [seq_len, num_heads, head_dim] 或 [seq_len * num_heads, head_dim]
//  ---- 契约： x最后一维必须等于head_dim(内部会断言)。
// num_heads: 注意力头数量
// head_dim: 单个头的维度（显示传入，绝不从shape反推 -- 见check）
// pos_offset: 起始位置偏移量（Prefill 为 0，Decode 阶段为当前 KV Cache 长度）
// base: RoPE 频率基数，通常为 10000.0f 或 500000.0f
//
// 约定约定约定（重要）：本实现采用 INTERLEAVED RoPE —— 按相邻成对的维度旋转
//   (x[2i], x[2i+1])，属于 RoFormer / GPT-NeoX 风格。
//   注意：HuggingFace Llama 使用的是 SPLIT-HALF 约定（rotate_half：前半段与后半段
//   交叉配对，即 (x[i], x[i + head_dim/2])），二者不同。
//   本项目内部自洽（PyTorch ref 亦为 interleaved），因此对拍一致；但将来若要
//   加载真实 Llama / HF 权重，需先将 Q/K 权重按对应关系做一次行置换，或改写为
//   split-half 实现，否则会静默产生错误结果（不报错，只是输出变乱）。
void rope_inplace(Tensor& x, int numHeads, int headDim, int posOffset = 0, float base = 10000.0f);
