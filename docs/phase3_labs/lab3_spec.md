# Phase 3 · Lab 3 - FP8 量化 · Stage A-D Coding Spec

> changelog: (无,首次落盘 2026-09-30, Lab3 只有一张知识卡 K1 已读完,故本 spec 直接吸收 K1 
内容,不走"读完知识卡再修订一次"的流程。)
> 一次性给全 4 个 Stage,不逐个解锁。目标语言 C++17,写在 `OperationNiobium_Lab`(WSL2 本机,CPU-only,无 CUDA/Blackwell 
硬件)。

## 0. 这个 Lab 在干什么

Lab 1 的 `PagedKVCache`(` kernels/paged_kv/include/paged_kv_cache.h`)把 K/V 存成 `std::vector<std::vector<float>> kPool_, vPool_`, fp32 每个元素 4 字节。Lab 3 把这层存储换成 **e4m3 FP8**(K1 读的 `torch.float8_e4m3fn`):1 字节/元素,内存直接砍到 1/4。

**警告澄清:这不是"手写推理 kernel"。铁律"推理热路径 kernel 不手搓,让 FlashInfer"管的是 attention 的矩阵乘 / 
softmax 这些算子。量化是**存储层**的 dtype 转换(quantize on write / dequantize on read),不产生新的推理数学,和 Lab 1 的 
`BlockAllocator` 是同一类"内存管理"活。所以这里可以手写。

**本机没有真 FP8 硬件**(`sm_120` / `CUTLASS 3.x` 这些是 K1 读的,不是本 lab 要跑的硬件路径),Lab 3 是**软件模拟 e4m3 编码** 
:自己写 quantize/dequantize,验证的是"量化引入多少误差、接口怎么嵌进现有 cache 层",不是硬件 kernel 性能。

**接口边界:** 只碰 `PagedKVCache` 这一层的存储 dtype。`BlockAllocator` / `BlockTable`(Lab 1)、`attention_paged` / 
`forward_paged`(Lab 2,还没写)都不动 — 量化对它们必须透明:调用方拿到的始终是 `float*`,量化/反量化在 `PagedKVCache` 
内部完成。

## 1. 公共约定

**e4m3 编码**(对齐 `torch.float8_e4m3fn`,K1 已读过的定义):

1 位符号 + 4 位指数(bias=7) + 3 位尾数
无 inf;S.1111.111 = NaN
最大有限值 = ±448
最小正规数 = 2^-6 = 0.015625;最小次正规 = 2^-9 ≈ 0.001953

**量化模型:静态 per-{layer, K|V} scale(PTQ 风格),不做 per-block/per-token 动态 scale。** 
理由:`write()` 在增量 decode 时每次只写 1 个 token,若 scale 跟着当前写入批次的 max-abs 动态算,同一个块会被不同 scale 
反复重新量化,语义复杂还慢。真实系统(vLLM FP8 KV-cache)也是校准阶段定一个 scale 就不再变。所以 scale 在 `PagedKVCache` 
`write()` 之前必须显式设好,之后整个生命周期不变。

```cpp
// kernels/paged_kv/include/fp8_e4m3.h —— 纯标量,不依赖 GPU / __nv_fp8
namespace paged_kv {

using Fp8 = uint8_t;

constexpr float kFp8MaxAbs = 448.0f;

// x/scale 舍入到 e4m3(round-to-nearest-even),越界 clamp 到 ±448。
// scale <= 0 是调用方 bug -> NB_CHECK.
Fp8 quantize_e4m3(float x, float scale);

// e4m3 位模式解码 * scale.
float dequantize_e4m3(Fp8 q, float scale);

} // namespace paged_kv
```

## 2. Stage A - 标量 quantize/dequantize(不碰 PagedKVCache)

目标: 先把 e4m3 编解码搞对、测对,再往 cache 里接。

接口

就是上面 fp8_e4m3.h 的两个自由函数。内部实现建议:
- quantize_e4m3:先 x / scale,clamp 到 [-448, 448],再按 e4m3 位域手算符号/指数/尾数(不要偷懒转 float 再 reinterpret 半精度库—K1 的重点就是搞懂这套编码,自己拆位)。
- dequantize_e4m3:拆符号/指数/尾数位,按公式 (-1)^s * 2^(e-7) * (1 + m/8) (e>=0,正规数)或 (-1)^s * 2^-6 * (m/8) (e==0,次正规)重建,再乘 scale。

验收判据
1. 已知值表对照:至少 8 个手算的 (float, scale=1.0) -> e4m3 位模式 对,覆盖:0、-0、最大有限值 448、最小正规 2^-6、次正规、一个需要舍入的中间值(验证 round-to-nearest-even,不是简单截断)。
2. round-trip 误差有界:随机(但确定性种子,不用 rand)采样 200 个 [-100, 100] 的浮点数,dequantize(quantize(x, 1.0), 1.0) 相对误差 <= 2^-3(e4m3 尾数 3 位的理论上界,约 12.5%)。
3. 不溢出 clamp,不抛不崩: quantize_e4m3(1e6, 1.0) 必须返回等于 quantize_e4m3(448.0, 1.0) 的编码,不是 NaN/未定义行为。
4. scale <= 0 -> NB_CHECK 抛 std::runtime_error。

## 3. Stage B - 接进 PagedKVCache

目标: kPool_ / vPool_ 从 vector<vector<float>> 换成 vector<vector<Fp8>>, write() / gather() / k_block() / v_block() 接口签名完全不变(仍收发 float*),量化/反量化在内部完成。

接口
class PagedKVCache {
public:
    PagedKVCache(int nLayers, int nKvHeads, int dHead, int nBlocks, int blockSize);

    // 新增:写入前必须为每层设一次 scale。设过之后不可再改(NB_CHECK 拦二次 set)。
    void set_scale(int layer, float kScale, float vScale);

    // 签名不变:内部 quantize 后存进 Fp8 池
    void write(int layer, const BlockTable& table, int startPos,
               const float* kSrc, const float* vSrc, int nTokens);

    // 签名不变:内部:从 Fp8 池 dequantize 到调用方的 float* 缓冲
    void gather(int layer, const BlockTable& table, int numTokens,
                float* kDst, float* vDst) const;

    // 返回值改动:裸块指针场景下不能再直接给 float*(池里是 Fp8),
    // 改成返回 dequantize 后写入调用方提供的 scratch buffer 的辅助函数:
    void k_block_dequant(int layer, BlockId b, float* dst) const;
    void v_block_dequant(int layer, BlockId b, float* dst) const;
    // 旧的 float* k_block()/v_block() 删除 — Lab 2 还没开工,没有调用方,不留兼容包袱。

private:
    std::vector<std::vector<Fp8>> kPool_, vPool_;   // 元素类型从 float 换成 Fp8
    std::vector<float> kScale_, vScale_;             // [nLayers],per-layer 静态 scale
    std::vector<bool> scaleSet_;                     // 防止漏 set_scale 就 write
    ...
};

write() 内部:对 [startPos, startPos+nTokens] 每个元素做 quantize_e4m3(src[i], kScale_[layer]) 存进 kPool_。NB_CHECK(scaleSet_[layer], ...) 放在最前面。

scale 从哪来:Stage B 的单测里手动指定(比如根据测试用的 genK/genV 生成函数算出 max-abs,再乘 1.05 留余量)。真实场景的"校准"(跑一遍 fp32 forward 统计 max-abs)不在本 lab 范围内,直接给定常量即可 — capstone 阶段再按校准流程。

验收判据
1. 复用 Lab 1 Stage B 的 5 组判据(bit-identical -> 改成"容差内一致",见下;跨块边界;两表交错不污染;越界抛异常;write 契约是 capacity 不是 num_tokens)。唯一改动:判据 1 从 "bit-identical" 改成 "dequantize 后与源值的相对误差 <= 2^-3(e4m3 精度上限,不是 bug)。
2. write() 在对应层 scaleSet_[layer] == false 时 -> NB_CHECK 抛异常,不静默用 0 或垃圾 scale。
3. 内存占用断言:同样的 (nLayers, nKvHeads, dHead, numBlocks, blockSize), sizeof 意义上 Fp8 池应为等价 float 池的 1/4(写个直接比较 kPool_[l].size() * sizeof(Fp8) vs ... * sizeof(float) 的断言即可,不用真的量内存)。

## 4. Stage C - 数值对拍:量化版 vs fp32 基线

目标: 证明"接了 FP8 之后,下游还能用"——这里下游还没有 Lab 2 的 attention,所以对拍对象是同一批 K/V 写入 + gather 出来做简单点积,不是完整 forward。

做法
1. 造一份确定性的 K/V 序列(复用 Stage A/B 的 genK/genV 风格生成函数),分别写入:
   - 一份 fp32 PagedKVCache(Lab 1 原版)
   - 一份 fp8 PagedKVCache(Stage B 新版,scale 用该序列的 max-abs 算)
2. 两份都 gather() 出来,算一个简单的 attention-like 量(比如 Q · K^T 用固定的 Q 向量,或者直接比较 gather 出的 K/V 本身)。
3. 比较 fp32 结果 vs fp8 结果的相对误差。

验收判据
1. gather 出的每个元素相对误差 <= 2^-3(继承 Stage A/B 的理论上界)。
2. 点积/norm 这类累加量的相对误差 <= 2^-2(累加会放大量化噪声,容差应该比单元素宽,不能直接套单元素的界 ——这条判据本身也是这个 lab 要验证的东西,别抄错公式)。
3. 至少覆盖两组不同的 scale(一组接近数据实际动态范围、一组明显偏小造成大量 clamp)——后一组应该能看到误差显著变大,证明测试真的对 scale 敏感,不是空跑。

## 5. Stage D - 边界 + 收尾

必做的边界组
1. scale 选太小:数据动态范围超过 scale * 448,大量值被 clamp → 断言"这种情况下误差会明显劣化"(不需要精确数字,断言"劣化组误差 > 正常组误差"即可),证明 Stage C 的两组对比不是巧合。
2. 全零输入:K/V 全 0 时 quantize/dequantize round-trip 应该精确为 0(不能因为浮点运算引入非零噪声)。
3. scale 未设就 write → Stage B 判据 2 已覆盖,这里补一条"先 set_scale 两次" → 第二次 NB_CHECK 拦下(前面 spec 里定的"设过不可再改")。
4. 跨层 scale 不串:kScale_[0] != kScale_[1] 时,两层各自量化解量化独立,不因为共享同一个 quantize_e4m3 自由函数就互相污染(理论上不会,但 Lab 1 Stage B 教训是"不可能失败的断言"这个坑,这条测试要真的让两层用不同到能测出差异的 scale)。

收尾
- 跑一遍 ctest,连同 Lab 1 的 14 个测试一起全绿,不能因为改了 PagedKVCache 的存储类型把 Lab 1 的东西弄挂(Lab 1 测试若还直接用旧的 float* k_block(),需要同步改成 k_block_dequant())。
- -Wall -Wextra 零新增警告。

## 6. Ruthless-trim 砍点(时间紧就往下砍,不要砍 Stage A/B)
- Stage D 第 4 组(跨层 scale 不串)—— 最先砍,理论上不太可能出错,加了主要是防"不可能失败的断言"的坑,不是核心验收。
- Stage C 第 3 组(两组 scale 对比)—— 其次砍,砍了就只验证"量化误差在界内",不验证"scale 选错会更差"这个更进一步的结论。
- Stage D 第 1/2 组 —— 不建议砍,这是这个 lab 唯一会在真实场景最踩坑的地方(scale 校准不准)。
- Stage A/B 不能砍 —— 砍了这个 lab 就没有验收了(和 Lab 1 spec 的第 6 节同一句话,同样的道理)。

## 7. 打勾规则

知识线 K1 已经打勾(读完即打)。代码线 Spec 这一项由 cron 自己落盘后打勾(本文件)。CODE 这一项(以及如果之后拆出 Stage 子项)只有 Runchen 亲自完成,明确说"做完了"或拉 git review 通过后才打勾 —— cron 只推进度卡,不代打、不越级。