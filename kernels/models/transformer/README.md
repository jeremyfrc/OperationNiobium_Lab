# C++ 手写 Transformer 前向 —— 全项目理解总结

> 这份文档不是代码说明，是"帮你把脑子里那团云变成结构"的复盘。
> 目标：读完之后你能自己讲出「每个文件为什么存在、每行代码在解决什么问题」。

---

## 0. 一句话概括你在做什么

你在**从零用 C++ 复刻一个现代小 LLM（Llama 系）的前向推理**，不训练、不反向传播。
验证方式不是"跑起来不崩"，而是**和一份自己用 numpy/PyTorch 写的参考实现逐张量对齐**。

这件事的本质是：
> 你在重建一个"数据从 token_id 一路流动到 logits"的**流水线**，
> 每一站（组件）都用一个已知正确答案去卡它，一站对不上就地爆炸，不往下走。

这就是 README 里那句 **"Correctness-first, perf 后置"** 的全部含义：
先在每站对拍，最后才谈快不快（Stage5 KV-cache 是 stretch）。

---

## 1. 正确的理解姿势：把它当"工厂流水线"看

整个模型是一条流水线，数据流是：

```
token_ids
  │  (embedding lookup：查表)
  ▼
[seq_len, d_model]  ← 这就是"活动的张量"，全程形状不变
  │
  ├── 重复 N 层（decoder layer）：
  │       x → RMSNorm → Attention → +x(残差) → RMSNorm → SwiGLU → +x(残差)
  │
  ▼
final RMSNorm
  │
  ▼
LM Head (matmul)
  │
  ▼
logits [seq_len, vocab_size]  ← 每个位置对"下一个 token"的打分
  │  (argmax / sample)
  ▼
next token  → 再喂回去（自回归 generate）
```

**关键心智模型**：除了开头（embedding）和结尾（logits），中间全程都是 `[seq_len, d_model]` 这个形状。
所有复杂组件（attention、FFN）都是"进去 `[seq_len, d_model]`，出来 `[seq_len, d_model]`"的黑盒。
只要牢牢抓住这一点，整个代码就不乱。

---

## 2. 为什么是这样分层/分文件：按"职责"切，而不是按"方便"切

README 里的文件布局不是随意的，它精确对应流水线的每一站：

| 文件 | 它负责的"一站" | 为什么独立 |
|---|---|---|
| `tensor.h/cpp` | 最底层的数据容器 | 所有算子都要用它，必须最先稳定 |
| `config.h` | 描述模型尺寸的一张表 | 尺寸是"契约"，所有函数都读它校验 |
| `ops.h/cpp` | 通用数学原语（matmul/add/softmax） | 是组件的地基，组件是它们的组合 |
| `rmsnorm.h/cpp` | 一站：归一化 | 单测最容易过，先把它钉死 |
| `rope.h/cpp` | 一站：位置编码 | 最容易写错，单独隔离方便对拍 |
| `attention.h/cpp` | 一站：注意力（含 GQA + causal mask） | 最复杂的组件，拆开单独验 |
| `swiglu.h/cpp` | 一站：前馈网络 | 另一块独立可测组件 |
| `transformer.h/cpp` | 把上面所有站"串"起来 | 只有组件都对了，串联才有意义 |

**为什么这么分？** 因为对拍策略是"分阶段、逐组件"：
- 如果全写在一个 `main.cpp` 里，出错时你根本不知道是 RoPE 错、还是 GQA repeat 错、还是 mask 错。
- 拆开后，**Stage1 四个单测各自独立**，谁红就是谁错。这就是"中间张量对拍能快速定位是哪个组件错"（README 第5节）的工程落地。

---

## 3. 逐个组件讲：为什么代码长这样

### 3.1 `Tensor` —— 为什么要有 `strides_`？

```cpp
float& Tensor::at(std::initializer_list<int> idx) {
    size_t flat_idx = 0;
    for (size_t i = 0; i < idx.size(); ++i)
        flat_idx += (*it) * strides_[i];   // 多维下标 → 一维下标
    return data_[flat_idx];
}
```

- 底层存储是**一段连续的 `std::vector<float>`**（行主序 / row-major）。
- 所谓 N 维张量，只是"连续内存 + 一套下标换算规则"。
- `strides_` 就是那套规则：第 i 维走 1 步，在平坦内存里跨多少个元素。

`reshape` 为什么只是改 `shape_` 和 `strides_`，不动 `data_`？
> 因为**行主序下，同一个 `numel` 的不同逻辑形状，物理内存布局完全一样**。
> 例如 `[8, 4, 4]`（seq, heads, head_dim）和 `[32, 4]`（seq*heads, head_dim）是同一段内存。
> 这正是 attention 里反复 reshape 的合法性来源（见 3.4）。

### 3.2 `matmul` / `softmax` / `add_inplace`（ops）

```cpp
for (i) for (k) { float a = A[i*K+k]; for (j) C[i*N+j] += a * B[k*N+j]; }
```

- 这是 **naive 三重循环 GEMM**，就是你 GEMM 那条线里 "naive-tiled" 的起点。
- `i-k-j` 循环顺序不是随便选的：它让 `B` 按行连续访问（cache 友好），这就是"先拍对，再谈优化"的最朴素版本。
- `softmax` 先减 max 再 exp，是为**数值稳定**（防止 `exp` 上溢）。这是所有 softmax 的标准写法。
- `add_inplace` 特意做成原地：残差连接 `x = x + f(x)` 天然是原地累加，省一次分配。

### 3.3 RMSNorm —— 为什么没有减均值、没有 bias？

```cpp
ms   = sum(x^2) / N;
inv  = 1/sqrt(ms + eps);
out[j] = x[j] * inv * weight[j];
```

对照 LayerNorm：LayerNorm 会 `(x - mean)/std * γ + β`。
RMSNorm **去掉了 mean 减法和 bias β**，只按"均方根"缩放。
- 为什么？这是现代 LLM（Llama）的简化选择：训练更快、效果不差。
- 这也解释了为什么它的单测比较好写、好对拍——公式短。

### 3.4 RoPE —— 最容易错的地方，也是你踩坑的地方

两块内容要分清：

**(a) 旋转公式**（每对相邻的两个元素 `(2i, 2i+1)` 一起旋转）：
```
角度 angle = 位置pos × 频率freq
freq_i = 1 / theta^(2i / head_dim)
x0' = x0·cos - x1·sin
x1' = x0·sin + x1·cos
```
这叫 **interleaved（交错式）RoPE**，`dump_reference.py` 里的 `rope_interleaved` 就是它的 numpy 版。

**(b) 位置 `pos` 从哪来 —— 这是你代码里那段注释在解释的坑：**

```cpp
int seqIdx = (r / numHeads) + posOffset;   // 第 r 行属于第几个 token
```

`rope_inplace` 要求输入最后一维必须是 `head_dim`。但你的投影输出是 `[seq_len, heads*head_dim]`，
如果直接丢进去，RoPE 会把**整行 `heads*head_dim` 当成一个巨大的 head** 去旋转，频率/位置全错。

所以 attention 里先 reshape：
```cpp
q_linear.reshape({seq_len * n_heads, d_head});   // [seq*heads, head_dim]
rope_inplace(...);                                // 每行 = 一个 (token, head)
q_linear.reshape({seq_len, n_heads * d_head});   // 还原
```
> `r / numHeads` 得到这一行是第几个 token（即真实的位置 pos），
> 这就是"为什么 row 内要除以 numHeads"。

`posOffset` 参数是给 KV-cache decode 用的：decode 时新 token 的绝对位置 = 已缓存 token 数 + 当前偏移（见 3.6）。

### 3.5 GQA Attention —— MHA 的超集

`head_flat(seq, head, d, n_heads, head_dim)` 这个内联函数是**整个 attention 的"单一真相源"**：
```cpp
return seq * (n_heads*head_dim) + head*head_dim + d;
```
所有手写下标统一走它 → 改 layout 只改一处，杜绝 `n_heads`/`n_kv_heads` 写反。

**GQA 的核心只有一件事**：`n_kv_heads < n_heads`，多个 Q head 共享一组 K/V。

```cpp
int kv_group_size = n_heads / n_kv_heads;
...
int kv_h = h / kv_group_size;   // 第 h 个 Q head 该用第几个 K/V head
```
- 当 `n_kv_heads == n_heads` 时 `kv_group_size==1`，退化成标准 MHA。
- 你的 tiny_cfg 设成 `4 heads / 2 kv_heads`，**强制走 GQA 路径**——README 自查清单专门提醒"别偷懒化成 MHA"。

**Causal mask**：
```cpp
if (j > i) { srow[j] = -1e9f; continue; }   // 未来位置不可见
```
把未来位置的分数打成 `-1e9`，softmax 之后近似 0，等于屏蔽。`tril` 下三角结构。

这些注释（RoPE reshape、GQA 的 `kv_h`、mask）之所以写这么细，是因为 README 第5节明确点名：
> "超了先怀疑 RoPE 角度计算或 GQA 的 K/V repeat 逻辑（这两个最容易错位）"

### 3.6 KV-Cache —— Stage5 stretch，但你已经做了

普通 `generate` 每生成一个 token 就把**整段序列重算一遍**，O(n²) 级浪费。
KV-cache 的想法：**K、V 一旦算过就不再重算**，缓存起来，新 token 只算自己的 Q，然后和缓存里全部历史 K/V 做注意力。

代码里的关键点：
```cpp
// prefill：整段 prompt 一次喂入，pos_offset = 0
transformer_forward_kv(prompt_ids, ..., pos_offset=0);

// decode：每次只喂 1 个新 token，pos_offset = 已缓存 token 数
std::vector<int> one = { next };
transformer_forward_kv(one, ..., pos_offset);
```

`attention_forward_kv` 相比 `attention_forward` 多了：
1. 把新算的 K/V **写进 cache**（`pos_offset + i` 位置）；
2. 注意力范围是 `[0, pos_offset+seq_len)`（历史 + 当前），而不是只看当前片段；
3. mask 变成"只能看 `j <= 绝对位置 abs_i`"。

> **验证方式很聪明**：`test_generate_kv.cpp` 不拿它跟 Python 比，而是跟**非 KV 版 `generate` 逐步比 logits**（rtol=1e-5）。
> 逻辑是：两条路算了同一件事，结果必须数值级一致。这比"只比 token id 一致"严格得多（token id 相同可能只是 argmax 恰好相同）。

---

## 4. 为什么用"对拍"而不是"记住正确答案"

这是本项目最重要、也最容易被忽略的**方法论**：

```
你的 C++ 实现   ←──比较──→   你自己写的 numpy/PyTorch 参考实现
（要验证的）                  （oracle / 权威答案）
```

- oracle 不是"抄一份标准答案"，是**你另写一遍**（`dump_reference.py`），固定随机种子，边算边 dump 中间张量。
- 为什么用二进制 `.bin`？直接 `tofile()` / `loadBinFile`，不依赖 JSON/文本精度损失，float32 逐字节对齐。
- 容差（`check_close`）：`diff <= atol + rtol * |expected|`，单组件 `rtol=1e-4, atol=1e-5`，整栈放宽到 1e-3。
  - 这不是"凑合"，而是**float32 累积误差的物理事实**——不同求和顺序结果就会差一点点。
  - 所以"对拍"检验的是**算法正确性**，不是"逐位相同"。

**分阶段对拍的价值**（README 第3节）：
| Stage | 对拍对象 | 通过标准 |
|---|---|---|
| S1 | 每个组件单独 | 误差在容差内 |
| S2 | 单 attention block | 含 mask 生效 |
| S3 | 整栈 logits | logits 对齐 |
| S4 | generate 输出 | token id 逐位一致 |
| S5 | KV vs 非KV | logits 数值级一致 |

---

## 5. 那些"看着多余"的设计，其实是在防坑

- **`NB_CHECK` vs `assert`**：`NB_CHECK` 不受 `-DNDEBUG` 影响，release 里依然生效——用来守 shape/layout/numel 这种"错了必须当场炸"的不变量。
- **入口全部按 `cfg` 校验形状**（attention 开头那一堆 `NB_CHECK`）：让错误**在入口爆炸**，而不是让一个形状错位的张量一路漂到输出，最后你对着 logits 误差发呆。
- **残差前先 `std::copy(x, out)`**：因为 `x` 后面可能还要用，不能原地改它。这是"以清晰换一点点拷贝开销"的取舍。
- **double buffer（`buffer_a`/`buffer_b` + `swap`）在整栈里**：避免每层都新建张量，且避免输入输出别名。
- **`head_flat` 单一真相源**：前面说了，防止 GQA/MHA 下标写反。

---

## 6. 一张图串起来（建议你照着默画一遍）

```
token_ids ──embed lookup──► x:[S,D]
                              │
        ┌───────── 每层 ↓ ─────┴──────────────┐
        │  x_norm1 = RMSNorm(x)                │
        │  q,k,v = x_norm1 @ Wq/Wk/Wv          │
        │  reshape → RoPE(q), RoPE(k) → 还原    │
        │  scores = q kᵀ / √d_head  (+causal)   │
        │  softmax(scores)                       │
        │  ctx = scores @ v  (GQA: kv_h=h/group) │
        │  attn = ctx @ Wo                       │
        │  x = x + attn          (残差1)         │
        │  x_norm2 = RMSNorm(x)                  │
        │  ffn = SwiGLU(x_norm2)                 │
        │  x = x + ffn           (残差2)         │
        └───────────────────────────────────────┘
                              │
                     final RMSNorm
                              │
                     logits = x @ lm_head  → [S, vocab]
                              │
                     argmax → next token → 喂回(自回归)
```

KV-cache 版唯一的区别：q/k/v 里的 **k、v 存进 cache**，decode 时 k/v 从 cache 读，位置用 `pos_offset`。

---

## 7. 你真正学到的东西（元层面）

1. **张量就是"连续内存 + 下标换算"** —— strides、reshape 不改数据，这是所有深度学习框架的底层真相。
2. **现代 LLM = RMSNorm + RoPE + GQA + SwiGLU + 残差** 五件套的堆叠，你以为很神秘的 Transformer，拆开就是这几站。
3. **正确性工程 = oracle 对拍 + 分阶段隔离 + 容差**。这是比"会写某段代码"更值钱的技能。
4. **性能优化是后置的**：先 naive 三重循环跑对（GEMM 那条线的翻版），再谈 tiling/SIMD/KV-cache。
5. **调试方法论**：形状错误在入口炸掉，组件错误用中间张量定位，别让误差在整栈里"滚雪球"。

---

## 8. 自查：你现在能回答这些问题吗？

- [ ] 为什么 attention 里 RoPE 前要 reshape 成 `[seq*heads, head_dim]`？不 reshape 会怎样？
- [ ] GQA 的 `kv_h = h / kv_group_size` 在 `n_kv_heads==n_heads` 时会退化成什么？
- [ ] 残差连接为什么要求输入输出形状完全一致 `[S,D]`？
- [ ] KV-cache 里 `pos_offset` 的物理含义是什么？为什么 decode 每步 +1？
- [ ] 为什么对拍用 `rtol + atol` 而不是"完全相等"？
- [ ] 为什么 `NB_CHECK` 要用抛异常而不是 `assert`？

如果这几题都能顺畅回答，这个 project 你就真的"做功"了，而不是"被 AI 拉着跑完"。
```
