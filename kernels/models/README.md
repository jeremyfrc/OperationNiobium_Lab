# C++ 手写 Transformer 前向 - Spec + 脚手架

## 1. 目标
从零写一个 **decoder-only Transformer 前向**(C++,无 autograd,无训练)—这就是 capstone 的 gen loop 前身。不抄 nanoGPT/llama.cpp 代码，可以对照理解结构，但实现自己写。
**Correctness-first, perf 后置。** 和 GEMM naive-tiled 的路子一样:先拍对、再谈优化(Stage5 = stretch,类比 GEMM 的 register tiling)。

## 2. 架构范围(对齐 CS336 L3)
decoder-only,现代小 LLM 结构(Llama 系一脉):
```
- token_ids
- token_embedding [seq_len, d_model]
- -> N llm layers:
    x_norm = RMSNorm(x)
    attn_out = Attention(x_norm) [RoPE + causal mask + MHA/GQA]
    x = x + attn_out (残差)
    x_norm2 = RMSNorm(x)
    ffn_out = SwiGLU_FFN(x_norm2)
    x = x + ffn_out (残差)
- final RMSNorm
- LM head (linear) [seq_len, vocab_size]
- (gen loop) sample / argmax next token, 自回归 decode
```
四个组件,都是 L3 讲过的:
-**RMSNorm**(替代 LayerNorm,无 mean-subtraction,无 bias)
-**RoPE**(旋转位置编码,作用在 Q/K 上,不用在 V)
-**MHA 或 GQA**(GQA = K/V head 数 < Q head 数,多个 Q head 共享一组 K/V;建议直接做 GQA,MHA 是 n_kv_heads = n_heads 的特例)
-**SwiGLU FFN**(gate-up 两路投影 + SiLU 激活相乘,再 down 投影;不是普通 ReLU-MLP)

### 2.1 Tiny config(对拍用,小到能手算/能塞进 numpy)
```cpp
// 仅供对拍测试;真实感知可以另开一个稍大 config,但先把这个跑通
TransformerConfig tiny_cfg {
  .vocab_size = 50,
  .d_model = 16,
  .n_layers = 2,
  .n_heads = 4,
  .n_kv_heads = 2, // 强制走 GQA 路径,别只测 MHA 特例
  .d_head = 4, // d_model / n_heads
  .d_ff = 32, // SwiGLU 隐藏维,tiny 阶段随便定,真实场景是 8/3*d_model 取整到 256 倍数
  .max_seq_len = 8,
  .rope_theta = 10000.0f,
  .rms_eps = 1e-5f,
};
```

约束(写进 stub 的注释里,不写 assert 逻辑,自己实现校验):`d_model == n_heads * d_head`, `n_heads % n_kv_heads == 0`.

## 3. 分阶段 build 计划 + 检查点
和 GEMM naive-tiled+(register tiling stretch)的路子一样,每个 Stage 有明确的"跑通"标准,别混着来。

-**Stage0 – 骨架**: `Tensor` 类型 + `matmul`/`add`/`softmax` 算子 + `TransformerConfig` struct + 全部文件骨架(头文件签名齐全,CPP 里全是 `TODO`)。**检查点**:能编译,能链接,不要求跑对。
-**Stage1 – 组件级单测**:RMSNorm / RoPE / softmax / SwiGLU **各自**对应 tiny numpy/PyTorch 参考(固定随机种子权重和输入,dump 中间张量);**检查点**:4 个组件测试各误差落在容忍内(建议 rtol=1e-4, atol=1e-5,float32 累积误差可以放宽到 1e-3 看情况)。
-**Stage2 – 单 attention block 前向**:RoPE + causal mask + GQA 组装成一个完整 attention block,对拍参考实现的 attention 输出。**检查点**:attention_forward 输出对拍过关,包括 causal mask 生效(未来位置不可见)。
-**Stage3 – 整层 -> 整栈前向**:decoder layer(RMSNorm->attn->残差->RMSNorm->SwiGLU->残差)对比,再是 N 层 + final norm + LM head,加载 tiny 权重做完整个 forward,和参考 logits 对比。**检查点**:tiny_cfg 下 2 层全通,logits 误差在容忍内。
-**Stage4 – gen loop**:自回归 decode(prefill + 逐 token generate,greedy argmax 先够用,采样可选)。**检查点**:给定 prompt token 序列,生成的 token id 序列和参考实现(相同权重、greedy)在 token 一致。
-**Stage5(后置,stretch)– perf/优化**:批处理、KV-cache(为 Phase 3 铺路)、内存复用、SIMD/多线程。**不阻塞前面阶段,正确性优先**。

## 4. 脚手架文件布局

```
models/transformer/
├── CMakeLists.txt
├── include/
│ ├── tensor.h # 极简 Tensor 类型(shape + 连续 float 存储)
│ ├── config.h # TransformerConfig struct
│ ├── ops.h # matmul / add / softmax 基础算子
│ ├── rmsnorm.h
│ ├── rope.h
│ ├── attention.h # MHA/GQA + causal mask
│ ├── swiglu.h
│ └── transformer.h # DecoderLayer / TransformerWeights / forward / generate
├── src/
│ ├── tensor.cpp # stub(声明已在头文件,实现留空 TODO)
│ ├── ops.cpp
│ ├── rmsnorm.cpp
│ ├── rope.cpp
│ ├── attention.cpp
│ ├── swiglu.cpp
│ └── transformer.cpp
└── tests/
    ├── ref/
    │ └── dump_reference.py # 接口骨架:同权重 tiny PyTorch/numpy 参考,dump 中间张量 + logits
    ├── test_rmsnorm.cpp
    ├── test_rope.cpp
    ├── test_softmax.cpp
    ├── test_swiglu.cpp
    ├── test_attention_block.cpp
    └── test_full_forward.cpp
```

### 4.1 `include/tensor.h`
```cpp
#pragma once
#include 
#include 

// 极简行主序张量,支持任意维 shape,Stage0 只要能编译;
// 内部存储/索引方式自己定,这里只固定对外接口形状。
class Tensor {
public:
    explicit Tensor(std::vector shape);

    float& at(std::initializer_list idx);
    const float& at(std::initializer_list idx) const;

    float* data();
    const float* data() const;

    const std::vector& shape() const;
    size_t numel() const;

private:
    std::vector shape_;
    std::vector data_;
};
```

### 4.2 `include/config.h`
```cpp
#pragma once

struct TransformerConfig {
    int vocab_size;
    int d_model;
    int n_layers;
    int n_heads;
    int n_kv_heads; // GQA: n_kv_heads out: [seq_len, d_model]
// 内部:Q/K/V 投影 -> RoPE(Q,K)-> GQA 分组重复 K/V -> causal mask ->
// softmax(QK^T / sqrt(d_head)) -> @V -> Wo 投影
void attention_forward(const Tensor& x,
                       const AttentionWeights& w,
                       const TransformerConfig& cfg,
                       Tensor& out);
```

### 4.3 `include/ops.h`
```cpp
#pragma once
#include "tensor.h"

// C[M,N] = A[M,K] @ B[K,N]
void matmul(const Tensor& A, const Tensor& B, Tensor& C);

// A += B,逐元素,shape 必须一致
void add_inplace(Tensor& A, const Tensor& B);

// 沿最后一维做 softmax,原地
void softmax_inplace(Tensor& x);
```

### 4.4 `include/rmsnorm.h`
```cpp
#pragma once
#include "tensor.h"

// out = x / rms(x, 沿最后一维) * weight
// rms(x) = sqrt(mean(x^2) + eps)
void rmsnorm(const Tensor& x, const Tensor& weight, float eps, Tensor& out);
```

### 4.5 `include/rope.h`
```cpp
#pragma once
#include "tensor.h"

// 对 x[形状 (seq_len, n_heads, d_head)]原地施加旋转位置编码
// position_offset:decode 阶段用于对齐 KV-cache 之后的绝对位置(Stage4 才用得上,
// Stage1-3 传 0 即可)
void apply_rope(Tensor& x, int position_offset, float theta);
```

### 4.6 `include/attention.h`
```cpp
#pragma once
#include "tensor.h"
#include "config.h"

struct AttentionWeights {
    Tensor Wq;  // [d_model, n_heads * d_head]
    Tensor Wk;  // [d_model, n_kv_heads * d_head]
    Tensor Wv;  // [d_model, n_kv_heads * d_head]
    Tensor Wo;  // [n_heads * d_head, d_model]
};

// x: [seq_len, d_model] -> out: [seq_len, d_model]
// 内部:Q/K/V 投影 -> RoPE(Q,K)-> GQA 分组重复 K/V -> causal mask ->
// softmax(QK^T / sqrt(d_head)) -> @V -> Wo 投影
void attention_forward(const Tensor& x,
                       const AttentionWeights& w,
                       const TransformerConfig& cfg,
                       Tensor& out);
```

### 4.7 `include/swiglu.h`
```cpp
#pragma once
#include "tensor.h"

struct FFNWeights {
    Tensor W_gate; // [d_model, d_ff]
    Tensor W_up; // [d_model, d_ff]
    Tensor W_down; // [d_ff, d_model]
};

// out = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down
void swiglu_forward(const Tensor& x, const FFNWeights& w, Tensor& out);
```

### 4.8 `include/transformer.h`
```cpp
#pragma once
#include 
#include "tensor.h"
#include "config.h"
#include "attention.h"
#include "swiglu.h"

struct DecoderLayerWeights {
    Tensor rms1_weight;
    AttentionWeights attn;
    Tensor rms2_weight;
    FFNWeights ffn;
};

struct TransformerWeights {
    Tensor token_embedding; // [vocab_size, d_model]
    std::vector layers;
    Tensor final_rms_weight;
    Tensor lm_head; // [d_model, vocab_size]
};

// 单层前向:x -> out,shape 均 [seq_len, d_model]
void decoder_layer_forward(const Tensor& x,
                           const DecoderLayerWeights& w,
                           const TransformerConfig& cfg,
                           Tensor& out);

// 整栈前向:token_ids -> logits [seq_len, vocab_size]
void transformer_forward(const std::vector& token_ids,
                         const TransformerWeights& w,
                         const TransformerConfig& cfg,
                         Tensor& logits);

// 自回归生成(Stage4):greedy argmax 版本先实现,采样可选 stretch
std::vector generate(const std::vector& prompt_ids,
                          const TransformerWeights& w,
                          const TransformerConfig& cfg,
                          int max_new_tokens);
```

### 4.9 `tests/ref/dump_reference.py`(接口骨架,不是可跑实现)

```python
"""
参考实现骨架:用 tiny PyTorch (或纯 numpy) 写一个功能等价 decoder-only
transformer,固定随机种子生成权重,dump 成 .npz 给 C++ 测试对拍。
不要在这里"顺手"把 C++ 该写的逻辑抄一遍再翻译—这个脚本本身也要自己写,
只是它不算"capstone 手写 C++"的一部分,可以用 PyTorch 高层 API 图快。

需要 dump 的产物(建议命名):
tiny_weights.npz # 所有权重,key 命名和 TransformerWeights 字段对应
ref_rmsnorm.npz # Stage1: 输入 x, weight, 输出 out
ref_rope.npz # Stage1: 输入 x(pre-rope), position_offset, theta, 输出 x(post-rope)
ref_softmax.npz # Stage1
ref_swiglu.npz # Stage1
ref_attention_block.npz # Stage2: 输入 x + attn weights, 输出 attn_out
ref_full_forward.npz # Stage3: 输入 token_ids, 输出 logits(每层中间态可选 dump 方便 debug)
ref_generate.npz # Stage4: 输入 prompt_ids + max_new_tokens, 输出 生成的 token id 序列(greedy)

TODO(Runchen): 用 tiny_cfg 参数实例化一个对应结构的 PyTorch nn.Module,
手动随机和 C++ 端的 weight 命名/shape 对得上,固定 seed,跑一遍 forward,
把中间张量和最终结果按上面文件名 dump 出来。
"""
```

### 4.10 测试 harness 骨架(以 `test_rmsnorm.cpp` 为例,其余同构)

```cpp
#include 
#include 
#include 
#include "rmsnorm.h"

// TODO: 引入自己的 npz 读取方式(cnpy / 手写简单 loader / 直接用文本 dump 避免依赖)

int main() {
    // TODO: 从 ref_rmsnorm.npz 读取 x, weight, expected_out
    // Tensor x = load_tensor(...);
    // Tensor weight = load_tensor(...);
    // Tensor expected = load_tensor(...);

    // Tensor out(x.shape());
    // rmsnorm(x, weight, /eps=/1e-5f, out);

    // TODO: 逐元素比较 out vs expected,容差 rtol=1e-4, atol=1e-5
    // for (size_t i = 0; i < out.numel(); ++i) {
    // float diff = std::fabs(out.data()[i] - expected.data()[i]);
    // float tol = 1e-5f + 1e-4f * std::fabs(expected.data()[i]);
    // assert(diff <= tol);
    // }

    std::cout << "test_rmsnorm: TODO not yet implementedn";
    return 0;
}
```

`test_attention_block.cpp` / `test_full_forward.cpp` 结构一样,只是加载的 npz 和调用的函数换成 `attention_forward` / `transformer_forward`;
`test_full_forward.cpp` 额外要跑通"拿找用 tiny_weights.npz 里的权重初始化 TransformerWeights"这一步(这部分是权重加载逻辑,也算 Stage3 工作里的一部分,自己写)。

### 4.11 `CMakeLists.txt`(stub)

```cmake
cmake_minimum_required(VERSION 3.16)
project(niobium_transformer_forward CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(transformer_forward
    src/tensor.cpp
    src/ops.cpp
    src/rmsnorm.cpp
    src/rope.cpp
    src/attention.cpp
    src/swiglu.cpp
    src/transformer.cpp
)

target_include_directories(transformer_forward PUBLIC include)

# TODO: 每个 Stage 起一个 test 可执行文件,逐个 add_executable + target_link_libraries
# add_executable(test_rmsnorm tests/test_rmsnorm.cpp)
# target_link_libraries(test_rmsnorm PRIVATE transformer_forward)
```

## 5. 对拍策略

- **客观实现** = 同一套 tiny 权重(固定 seed)在 PyTorch/numpy 里跑的等价 forward,不是"标准答案抄过来",是自己另写一遍高层来实现 oracle。
- 每个 Stage 对比:npz中间张量**(Stage1-2 组件级)** -> **最终 logits / 生成 token**(Stage3-4)。中间张量对拍能快速定位是哪个组件错,别只看最终 logits—就算全往前推,积也会跟着雪。
- 容差建议:单组件 rtol=1e-4, atol=1e-5(float32);累积到整栈 forward 可以放宽到 1e-3 量级,如果超了先怀疑 RoPE 角度计算或 GQA 的 K/V repeat 逻辑(这两个最容易错位)。
- Stage4 的生成对拍用 **greedy argmax**(稳定性,不引入采样随机性),token id 序列要求逐位完全一致。

## 6. 自查清单(开写前过一遍)
- [ ] d_model == n_heads * d_head` 校验输写了吗(哪怕只是个 assert)
- [ ] GQA 的 K/V repeat 是不是真的跑到了(tiny_cfg 里 n_kv_heads=2 <= n_heads=4,别偷懒化成 MHA 测试)
- [ ] causal mask 是否真的挡住了未来 token(建议 Stage2 专门加一个"空一下未来 token 的值,验证输出不变"的测试)
- [ ] RoPE 的 position_offset 参数没设(Stage1-3 传 0,Stage4 才用得上)
- [ ] 中间张量 dump 的文件
