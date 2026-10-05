# Phase 3 · Lab 2 - FlashAttention-lite · Stage A-D Coding Spec

一次性给全 4 个 Stage,不逐个解锁。落盘 2026-10-02。接着 Lab 1 的 OperationNiobium_Lab (WSL2 本机), C++17.
changelog: (空 — K2/K3 读完后允许在这里记一次修订)

0. 这个 Lab 在干什么

Lab 1 Stage C/D 量出来一个结论(见 phase3-schedule.md 底部的 benchmark 定论): attention_paged_forward 每一步都要 cache.gather_into_scratch() 把这个序列到目前为止全部历史 K/V 拷成连续缓冲,再调 attention_over_kv() 算一次材料化的 [n_heads, seq_len, total_ctx] 完整 scores。这次拷贝不产生任何算术,纯带宽开销,且每解码一步就要重拷一次全历史。

Lab 2 把这次材料化拿掉:直接在物理块上算,用 online-softmax 维护 running max / running sum, K/V 只读一遍,不拷贝、不落一个完整的 scores 矩阵。 这就是 FlashAttention 论文(K1 已读)的核心递推,应用到 Lab 1 已经焊好的分页结构上。

铁律不改变含义,但这里特殊: 手写这个 kernel 不违反"推理热路径不手搓"——这是学习性,目的是亲手焊一遍 online-softmax 的递推,搞懂它为什么能把两趟扫描压成一遍。capstone 真正跑起来的那一层最终会换成 FlashInfer。Lab 2 产出的代码不会被当成生产路径去优化到底,它的价值在于"写过一次就懂了",写完之后再 README 里记一句这层和 FlashInfer 的关系就够了。

和 Lab 1 的分工边界: Lab 1 的 BlockAllocator / BlockTable / PagedKVCache 原样复用,一行不改。Lab 2 只新增一条"怎么读这些块来算 attention"的路径,和现有的 attention_paged_forward (gather 版)并存,不覆盖它——Stage D 要靠这两条路径互相对拍。

1. 公共约定(复用 Lab 1 + Phase 2 已有的)

using BlockId = int32_t;

测试配置,两套都要跑:

// 小到能手算,复用 Lab 1 Stage A-D 的配置
block_size = 4, num_blocks = 8, n_layers = 2, n_kv_heads = 2, d_head = 4, n_heads = 2 (MHA, kv_group=1)
// GQA 配置,复用 Phase 2 tiny_cfg(tests/ref/dump_reference.py)
n_heads = 4, n_kv_heads = 2, d_head = 16 (kv_group = 2)

两套都测的原因:第一套块小、好手算边界;第二套是唯一验证 kv_h = h / kv_group 映射有没有在新代码里写反的场景—Lab 1 从没测过 GQA,Lab 2 第一次直接在 attention 数学里碰 GQA,容易在这里翻车。

复用: Tensor NB_CHECK check_close() head_flat() TransformerConfig AttentionWeights paged_kv::PagedKVCache / BlockTable(只读,不改)、attention_over_kv() (Stage A/D 的对拍基准)、attention_paged_forward() (Stage D 的第二个对拍基准)。

块内数据布局不变(Lab 1 定的):

block 内: [block_size, n_kv_heads, d_head] 行优先
k_block(layer, b) / v_block(layer, b) 返回这一块的起始指针,长度 block_sizen_kv_headsd_head


以下是为您提取的图片中的所有文字：

## 2. Stage A - naive baseline:按块读,不经 gather

**目标:** 把"按物理块读数据"和"后面要改的 softmax 算法"这两件事分开调试。Stage A **数学上和现在的** `attention_over_kv` **完全一样**(还是材料化完整 `[n_heads, seq_len, total_ctx]` scores),唯一区别是不再调 `cache.gather_into_scratch()` 拷一份连续缓冲,而是直接从 `cache.k_block()/v_block()` 按块地址读。

### 接口

cpp
// kernels/models/transformer/include/attention_flash.h

// Stage A:朴素 baseline,直接按物理块读,仍 materialize 完整 scores。
// 数值必须与 attention_over_kv(gather 版)逐位 bit-identical —— 这是本 stage 的唯一判据。
void attention_over_paged_kv_naive(const Tensor& qRoped,
                                  const paged_kv::PagedKVCache& cache,
                                  const paged_kv::BlockTable& table,
                                  Tensor& ctx, const TransformerConfig& cfg,
                                  int layer_idx, int seq_len,
                                  int pos_offset, int total_ctx);

**写法提示:** 把 `attention_over_kv()` 里 `kc[j * kv_stride + kv_h * d_head + d]` 这一行的寻址换成"先定位 `(block_id, offset) = table.locate(j)`,再从 `cache.k_block(layer_idx, block_id)` 里按 `offset * kv_stride + kv_h*d_head + d` 取值"。外层循环结构(`h -> i -> j -> d` 四层)原样照抄,**先不要跳过已经知道用不到的块**——那是优化,放到 Stage B/C。

**会踩的坑:**
- `table.locate(j)` 对每个 `j` 都调一次,四层循环里 `j` 是最内层之一,调用开销会比 Lab 1 Stage B 的"整块拷"更高。**这是预期的,Stage A 本来就不是性能版本**,Stage D 的对拍只验证数值,不验证 Stage A 自己的速度。
- `j` 永远不会超过 `pos_offset + seq_len - 1`(= `total_ctx - 1`),所以 `locate(j)` 用到的块号永远 `<= table` 里已分配的块——不会读到未分配的块。但如果你把循环上界写错(比如用 `table.capacity_tokens()` 而不是 `total_ctx`),就会读到已分配但逻辑上还没写入的块,数值是垃圾但不会崩溃,**对拍才会抓到这个错**,review 代码时自己先查一遍循环上界。

### 验收判据

- [ ] 对 `total_ctx` 分别为 `block_size` 的整除(如 16)和不整除(如 11、23)两种情况,`attention_over_paged_kv_naive` 与 `attention_over_kv`(经 `cache.gather_into_scratch()` 喂同样数据)结果 **bit-identical**(两者浮点运算顺序完全相同,只是取数据的路径不同,必须位级相等,不是 `rtol`)
- [ ] GQA 配置(`n_heads=4, n_kv_heads=2`)下跑一遍,同样 bit-identical
- [ ] 循环上界用 `total_ctx`,不是 `table.capacity_tokens()`(写个专门的测试:`num_blocks` 比 `total_ctx` 需要的多分配 2 块,确认多出来的块不被访问——可以在测试里把多余块的内存填 `NaN`,跑完检查输出没有 `NaN`)


3. Stage B - tiling:两遍扫描,不materialize 完整 scores

**\目标:**** 去掉 Stage A 里那个 [n_heads, seq_len, total_ctx] 的完整 scores Tensor,改成按物理块一块一块读,中间缓冲只有 block_size
那么大。**\先用两遍扫描法**** (FlashAttention 论文里 online softmax 之前的中间形态):第一遍只为了拿到每行的全局 max,第二遍才算 exp
和加权。这一步还没把两遍合成一遍——那是 Stage C 的活,这里先把"分块读、不再整体材料化"这件事独立验证过。

接口

// Stage B:两遍扫描,中间缓冲 O(block_size),不是 O(total_ctx)。
void attention_over_paged_kv_tiled(const Tensor& qRoped,
                                  const paged_kv::PagedKVCache& cache,
                                  const paged_kv::BlockTable& table,
                                  Tensor& ctx, const TransformerConfig& cfg,
                                  int layer_idx, int seq_len,
                                  int pos_offset, int total_ctx);

**\算法形状(每个 head、每个 query 行 i 独立):****

abs_i = pos_offset + i
last_block = abs_i / block_size // 只访问到这一块,不多读

// Pass 1:求 row max(不存任何 score)
m = -inf
for b in 0..=last_block:
    (ptr, valid_len) = 这块里 j 的范围(最后一块只到 abs_i % block_size)
    for j_in_block in 0..valid_len:
        s = dot(q_i, k_block[j_in_block]) * scale
        m = max(m, s)

// Pass 2:用全局 max 算 exp-sum 和加权 V(这次才用一个 block_size 大小的小缓冲存当前块的 score)
l = 0; acc[d_head] = {0};
for b in 0..=last_block:
    local_scores[block_size] // 复用,别每块都 new
    for j_in_block in 0..valid_len:
        local_scores[j_in_block] = exp(dot(q_i, k_block[j_in_block]) * scale - m)
        l += local_scores[j_in_block]
    for j_in_block in 0..valid_len:
        acc[i] += local_scores[j_in_block] * v_block[j_in_block]
ctx[i] = acc / l

**\会踩的坑:****
- Pass 1 和 Pass 2 都要重算一次 dot(q_i, k_j)——这是故意的,Stage B 的卖点就是"用两遍读换掉一次整体材料化",不是省计算。省计算要等 Stage C 把两遍合一。
- local_scores 缓冲必须声明在 i(query 行)循环外面复用,不要每个 i 都重新分配——不是正确性问题,但如果写成每行 new,Stage D 的 benchmark 会被这个噪声盖住真正的信号。
- 最后一块的 valid_len 算错是这个 stage 最容易翻的地方:valid_len = (abs_i % block_size) + 1,不是 block_size。拿 total_ctx 不整除 block_size 的配置(比如 11)测,如果这里错了,输出会多算几个不存在的未来 token,数值会偏。

验收判据

- [ ] 与 Stage A(attention_over_paged_kv_naive)结果一致, rtol=1e-6 (允许因为两遍读和顺序和一遍 materialize 不完全相同导致的浮点误差,但量级应该极小,不是 1e-5 那种"数值对拍"级别的容差)
- [ ] 中间缓冲只有 block_size 量级——review 时自查代码里没有任何 new / Tensor({...total_ctx...}) 这种随 total_ctx 增长的临时对象
- [ ] total_ctx 不整除 block_size(11, 23)时最后一块的 valid_len 正确——专门写一个测试,人工算出 11 个 token 时最后一块应该只看 3 个
- [ ] (可选自检,不是硬判据) 每块 dot 计算确实被算了两遍——加一个调试计数器验证,通过后可以删除或用 #ifdef NB_DEBUG_COUNT 包起来


## 4. Stage C - online softmax:单遍扫描,K/V 只读一次

**目标:** 这是整个 lab 的核心。把 Stage B 的两遍合成一遍——用 FlashAttention 的 rescale 递推,让 Pass 1 的 max
"一边算一边追"而不是"算完再回去用"。

### 接口

cpp
// Stage C:online(单遍)softmax,FlashAttention 的核心递推。K/V 每块只读一遍。
void attention_over_paged_kv_flash(const Tensor& qRoped,
                        const paged_kv::PagedKVCache& cache,
                        const paged_kv::BlockTable& table,
                        Tensor& ctx, const TransformerConfig& cfg,
                        int layer_idx, int seq_len,
                        int pos_offset, int total_ctx);

// 入口:和 attention_paged_forward 同款投影/RoPE/cache.write,
// 只是最后一步从 "gather_into_scratch + attention_over_kv" 换成上面这个。
void attention_paged_forward_flash(const Tensor& x, const AttentionWeights& w,
                        const TransformerConfig& cfg, Tensor& out,
                        paged_kv::PagedKVCache& cache,
                        paged_kv::BlockTable& table,
                        int layer_idx, int pos_offset);

**递推(每个 head、每个 query 行 `i` 独立维护三个 running 量):**

m = -inf // running max
l = 0 // running sum(对齐到当前 m 的 exp 和)
acc[d_head] = {0} // running 加权 V(同样对齐到当前 m)

for b in 0..=last_block:
    local_max = 这块里 valid 的 score 的最大值 // 块内小循环,块内先求一次 max
    new_m = max(m, local_max)
    correction = exp(m - new_m) // m 第一次是 -inf 时 correction 必须是 0,见下面的坑
    l = l * correction
    acc[:] = acc[:] * correction
    for j_in_block in 0..valid_len:
        p = exp(score_j - new_m)
        l += p
        acc[:] += p * v_block[j_in_block]
    m = new_m

ctx[i] = acc / l

**会踩的坑(这是本 lab 真正的难点,出 bug 大概率在这几处):**

1. **m 初始为 -inf 时的第一块递推。**
    `correction = exp(-inf - new_m)`,数学上应该是 `0` (因为还没有任何历史要保留)。
    `std::exp(std::numeric_limits<float>::-infinity())` 在 IEEE 754 下确实算出 `0.0f`,**但不要依赖这个隐式行为过关**——显式判断 `if (m == -inf) { l = 0; acc 清零; }` 再进入正常递推,或者把 m 的哨兵值设成一个有限的极小数(比如 `-1e30f`)而不是真 `-infinity`,避免 `-inf - (-inf) = NaN` 这种边界(当 new_m 也是 -inf 时,比如这一步的所有 score 都是 -1e9f 被 causal mask
    掉——理论上不会发生,因为每行至少能看到自己这个 token,但写个断言防一下)。

2. **rescale 顺序。** 必须先用旧 `m` 和 `new_m` 算出 `correction`,**用它去缩放已经累积的 `l` / `acc`,再累加这一块的新贡献**。如果先更新
    `= new_m` 再算 `correction` ,`correction` 永远是 `exp(0)=1`,等于没做 rescale,结果会安静地错(不会崩、不会 NaN,只是数值偏——这种 bug
    最难查,Stage D 的数值对拍是唯一抓手)。

3. **GQA 映射。** `kv_h = h / kv_group` 这行从 Stage A 抄过来就行,但 online 递推是"每个 head 独立维护 `m`/`l`/`acc` ",容易在重构时不小心把 `m`
    /`l`/`acc` 声明成按 `kv_h` 索引而不是按 `h` 索引——GQA 下多个 `h` 共享同一个 `kv_h`,但**每个 `h` 的 Q 不同,running 统计量必须按 `h` 独立**
    才行。这个只有 GQA 配置(`n_heads=4, n_kv_heads=2`)能测出来,Stage A/B 如果只用 `n_heads=n_kv_heads` 的配置测过,这个坑会被完美掩盖到 Stage C
    token,`last_block` 可能等于 0(只有一块),递推的"第一块"分支必须正确处理只有一块就结束的情况。

### 验收判据

- 与 Stage A/B 结果 `rtol=1e-5` 内一致(同 Lab 1 Stage D 的容差口径)
- K/V 每块只读一遍——加调试计数器验证(同 Stage B 的可选自检,这里从"可选"变"必须",因为"只读一遍"是 Stage C 存在的全部意义)
- GQA 配置(`n_heads=4, n_kv_heads=2`)跑一遍,确认 3坑 没有翻车
- `total_ctx <= block_size` (只有一块)的 decode 场景专门测一遍,确认 $坑 1 的边界没有产生 `NaN`
- 一次性跑 `prompts=3 + 20` 步 decode 的完整序列(复用 Lab 1 Stage C 的测试结构),每一步都对拍,不只是测最后一步


5. Stage D - 全链路对拍 + benchmark

目标: 证明 attention_paged_forward_flash 可以直接替换 attention_paged_forward 用在真实的 prefill/decode 循环里,并且量出来它确实解决了 Lab 1 发现的那个问题。

必做的四组测试

1. 数值对拍(主判据) - attention_paged_forward_flash vs attention_paged_forward(gather 版)vs Phase 2 的 attention_forward_kv(连续 KV 版),三者在同一个 prompt+decode 序列上逐步对拍, rtol=1e-5.
2. Greedy 采样一致 - 三条路径采出的 token id 序列完全相同(复用 Lab 1 Stage C 的判据)。
3. 碎片化复用 - 直接复用 Lab 1 Stage D 第 2 组(3 序列 5/11/3 交错分配)的测试夹具,换成跑 attention_paged_forward_flash ,确认多序列场景下每个序列只读自己的 table/块,互不污染。
4. Benchmark - attention_paged_forward_flash vs attention_paged_forward,在 total_ctx = 20/50/100/250(和 Lab 1 benchmark 同样的点位,方便直接比)测单步 decode 耗时,warmup + median(别重犯 Lab 1 Phase 2 那个"只有一侧加 warmup"的错)。

验收依据

- [ ] 四组全绿
- [ ] benchmark 数字记进 benchmarks/,README 里写一句结论:flash 版有没有像预期那样甩掉 Lab 1 量出来的那条"全历史 gather 拷贝"曲线(20~250 token 时 gather 版是 2.45~163.53 μs ,flash 版预期不再随 total_ctx 近似二次增长,而是接近线性——如果测出来不是这样,说明 Stage C 哪里还在偷偷做整体拷贝,回去查)
- [ ] fresh clone + clean build 零 warning(保持 Lab 1 Stage D 收口时的水位)
- [ ] attention_paged_forward(gather 版)代码不删——它是以后任何人怀疑 flash 版算错时的对拍基准,留着

6. Ruthless-trim 砍点

可以砍:

1. Stage B(两遍扫描)可以跳过,直接从 Stage A 跳到 Stage C——但跳了就损失"亲手对比两遍 vs 一遍"这个机制理解的机会,K1/K2 论文讲的正是在这个对比,砍之前想清楚。如果砍,在 README 写一句为什么。
2. Benchmark 的点位可以只测 1-2 组(比如只测 50 和 250),不用复刻 Lab 1 全部 4 组。
3. Stage D 第 3 组(碎片化)如果 Stage A-C 单序列测试已经很扎实,可以简化成直接跑一遍而不额外加断言,不用重新设计夹具。

不能砍:

- Stage C 的单遍在线递推本身——这是这个 lab 存在的全部理由
- 块边界上的 causal mask 正确性(valid_len 算对)——paged + flash 结合最容易犯错的地方就是这里
- GQA 配置下的测试——Stage C 坑 3 那种错只有 GQA 配置测得出来
- Stage D 第 1 组(三路对拍)——砍了整个 lab 没有验收

7. 打勾规则

phase3-schedule.md 里 Lab 2 的 Spec 这一项,我自己打 [x](这是我的交付)。下面四个 Stage,只有你亲口说"做完了",或者拉 git 到主 space 让我 review 过,我才打 [x], cron 每天推的是进度卡,不代打、不越级。