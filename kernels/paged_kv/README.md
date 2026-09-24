# Phase 3 · Lab 1 – Paged KV-Cache · Stage A–D Coding Spec

> 一次性给全 4 个 Stage,不逐个解锁。落盘 2026-09-11。目标语言 `C++17,写在 `OperationNiobiumLab`(WSL2 本机)。

---

## 0. 这个 lab 在干什么

Phase 2 的 KV-cache 是**每序列一整块连续内存**: `k_caches[layer]` 形状 `[max_seq_len, n_kv_heads, d_head]`,按 `pos_offset` 线性索引。它能跑,但有三个死穴:

1. **必须按 `max_seq_len` 预留** —— 序列只用 20 个 token 也占 8192 个位置的内存
2. **不能共享** —— 两个序列有相同前缀也要各存一份
3. **不能局部释放** —— 要么整块在,要么整块没

Lab 1 把它换成 **分页(paged)**:内存切成固定大小的 block,一张 block table 把逻辑 token 位置映射到物理块。这是 vLLM 的核心机制,你 K2/K3 两张知识卡读的就是它。

**⚠️ 这不是一个练完就扔的 lab。** 按 2026-09-01 的收敛模型,Lab 1 写出来的 `BlockAllocator` / `BlockTable` **直接长成 AgentKV capstone 的地基**:

| Lab 1 里的东西 | 在 capstone 里变成 |
|---|---|
| `refcount` / `pin()` / `unpin()` —— 应用层锁住不许回收的 KV | |
| `decref` 归零的处理 | `evict_span()` —— 可回收但内容仍有效的池 |
| `BlockTable` 快照 | `checkpoint` / `restore` —— headline demo B(backtrack) |

所以下面有两处**必须留手**,写死了就等于把 capstone 的路堵上。我在 Stage A 里用 ⚠️ 标出来了。

**铁律不变:** 推理热路径的 kernel 不手搓,站 FlashInfer。Lab 1 是**内存层**,不碰 attention kernel —— Stage C 允许"gather 成连续再复用 Phase 2 的 attention"这条捷径,那正是设计不是偷懒。

---

## 1. 公共约定

```cpp
using BlockId = int32_t;
constexpr BlockId kInvalidBlock = -1;
```

**块内数据布局**(和 Phase 2 保持一致,`head_flat()` 可直接复用):

```
block 内: [block_size, n_kv_heads, d_head]   行优先
物理池:   k_pool[layer] = float[num_blocks * block_size * n_kv_heads * d_head]
```

**测试配置**(小到能手算):

```
block_size = 4, num_blocks = 8, n_layers = 2, n_kv_heads = 2, d_head = 4
```

复用 Phase 2 已有的:`Tensor`、`NB_CHECK`、`check_close()`、`head_flat()`、`tiny_cfg`。
(顺手把 `include/check.h` 末尾那个缺的换行补上,别再看了 4 条 warning。)

---

## 2. Stage A – BlockAllocator + BlockTable

**目标:** 只做内存簿记,不碰任何 K/V 数据。

### 接口

```cpp
class BlockAllocator {
public:
    BlockAllocator(int num_blocks, int block_size);

    BlockId allocate();                 // 无空闲 → kInvalidBlock(不抛异常)
    void    incref(BlockId);
    void    decref(BlockId);            // 归零 → 交给 on_zero_ref 策略,见下
    int     refcount(BlockId) const;

    int num_free() const;
    int num_blocks() const;
    int block_size() const;

private:
    std::vector<int> refcount_;         // [num_blocks]
    std::vector<BlockId> free_list_;
    int block_size_;
};
```

### ⚠️ 留手 1 — refcount 归零 ≠ 立即 free

> 最自然的写法是 `if (--refcount_[b] == 0) free_list_.push_back(b);`。**别这么写死。**

> capstone 需要第三种状态:*refcount 归零、但内容仍然有效、可被 evict 也可被复活*(prefix cache 就靠这个)。所以归零时走一个可替换的分支:

```cpp
enum class ZeroRefPolicy { FreeImmediately, MoveToEvictable };
```

> Stage A **只实现 `FreeImmediately`**;`MoveToEvictable` 留个 `NB_CHECK(false, "not implemented")` 即可。但**数据结构里必须给 evictable 池留位置**(一个 `std::list<BlockId>` 或类似 evictable_ 加一个 `BlockId → 迭代器` 的索引),别等以后再回来改结构。

> 判断标准:以后加 evict 时,应该只改策略分支,不改 `BlockAllocator` 的成员布局。

```cpp
class BlockTable {
public:
    explicit BlockTable(BlockAllocator* alloc);
    ~BlockTable();                       // 对所有持有块 decref

    bool ensure_capacity(int num_tokens);      // 按需 allocate;不够返回 false 且不留半成品
    std::pair<BlockId,int> locate(int token_pos) const;  // → (block_id, offset_in_block)
    void append_tokens(int n);            // 逻辑长度 += n(调用前须 ensure)

    int num_tokens() const;
    const std::vector<BlockId>& blocks() const;

    BlockTable clone_shared() const;      // ⚠️ 留手 2,见下

private:
    BlockAllocator* alloc_;
    std::vector<BlockId> blocks_;
    int num_tokens_ = 0;
};
```

### ⚠️ 留手 2 — BlockTable 必须可拷贝/可快照

> `clone_shared()` 拷贝 `blocks_` 这个**块号列表**,并对每个块 `incref()` —— **不拷贝任何 K/V 数据**。两张表从此指向同一批物理块,refcount 都是 2。

> 这是 capstone 里 `checkpoint()` 的原型,也是 headline demo B(backtrack)的全部机制。写起来只有五行,但如果 Stage A 把 `BlockTable` 做成不可拷贝的(比如 `unique_ptr` 持有块),后面就得整个重写。

> Stage A **不需要**实现写时复制(COW 分裂) —— 那是 capstone 的话。这里只要 `incref` 语义正确。

### 验收判据

- [ ] 分配满 8 块后 `num_free()==0`,再 `allocate()` 返回 `kInvalidBlock`(不崩、不抛)
- [ ] `decref` 到 0 → 块回 free list,`num_free()` 恢复
- [ ] `block_size=4` 时: `locate(0)==(b0,0)`, `locate(3)==(b0,3)`, `locate(4)==(b1,0)`, `locate(7)==(b1,3)`
- [ ] `ensure_capacity` 失败时**不能留下半分配状态**(要么全成功要么全回滚)
- [ ] `clone_shared()` 后每块 `refcount()==2`;原表析构后克隆表的 `locate()` 仍然有效
- [ ] 所有表析构后 `num_free() == num_blocks`(零泄漏)

---

## 3. Stage B – Cache 读写

**目标:** 有了簿记,现在真把 K/V 塞进块、再取回来。

```cpp
class PagedKVCache {
public:
    PagedKVCache(int n_layers, int n_kv_heads, int d_head,
                 int num_blocks, int block_size);

    // 把逻辑位置 [start_pos, start_pos+n) 的 K/V 写进 table 指向的块
    void write(int layer, const BlockTable& table, int start_pos,
               const float* k_src, const float* v_src, int n_tokens);

    // 把逻辑位置 [0, num_tokens) 的 K/V 收集成连续缓冲
    void gather(int layer, const BlockTable& table, int num_tokens,
                float* k_dst, float* v_dst) const;

    // 裸块指针(Stage C / 将来 Lab 2 用)
    float*       k_block(int layer, BlockId b);
    const float* k_block(int layer, BlockId b) const;
    float*       v_block(int layer, BlockId b);
    const float* v_block(int layer, BlockId b) const;

private:
    std::vector<std::vector<float>> k_pool_, v_pool_;  // [n_layers][num_blocks*block_size*n_kv_heads*d_head]
    int n_kv_heads_, d_head_, block_size_;
};
```

**写的时候会踩的坑:** `start_pos` 未必落在块边界上。一次 `write(n_tokens)` 通常要拆成 3 段:当前块的剩余尾巴 → 若干整块 → 最后一块的头。把这个拆分逻辑单独抽成一个函数,Stage C 还要用。

### 验收判据

- [ ] 写 17 个 token(`block_size=4` → 跨 5 块)再 `gather`,结果与源 **bit-identical**
- [ ] 跨块边界正确 —— 第 4 个和第 5 个 token 分属不同块,值不串
- [ ] 两个 `BlockTable` 用同一 allocator,**交错分配**(A 拿 2 块 → B 拿 2 块 → A 再拿 2 块),各自 `gather` 出来的数据互不污染。**这条是分页的核心价值,别跳过**
- [ ] `write` 越过 `table.num_tokens()` 能覆盖的范围 → `NB_CHECK` 拦住

---

## 4. Stage C – 增量 decode

**目标:** 接通真实的 prefill / decode 循环,和 Phase 2 的 `Transformer` 打通。

```cpp
// 每步(prefill n 个 / decode 1 个)统一入口:确保容量 → 写入 → 推进逻辑长度
bool append_kv(PagedKVCache& cache, BlockTable& table, int layer,
               const float* k_new, const float* v_new, int n_new);

// paged attention(Stage C 版:gather → 复用 Phase 2 的 attention)
void attention_paged(const Tensor& q, const PagedKVCache& cache,
                     const BlockTable& table, int layer,
                     Tensor& out, int n_heads, int n_kv_heads, int d_head);
```

**明确允许的捷径:** `attention_paged` 先 `gather()` 成连续 K/V,再直接调 Phase 2 的 `attention_forward_kv`。

**这不是偷懒,是分层。** Lab 1 只负责"内存怎么组织",attention 怎么在分页上直接算(不 gather)是 **Lab 2** 的事,而在 capstone 里那一层最终会换成 FlashInfer。现在手搓 `paged attention` kernel 既踩铁律,又会让 Stage D 的对比同时验两件事,出问题分不清是谁的锅。

**代价要记下来:** gather 每步都拷一遍全部历史 KV,所以 Stage C 的 `paged 路径`**会比 Phase 2 的连续 KV 慢**。这是预期内的。Stage D 要把这个数字量出来 —— 它就是 Lab 2 存在的理由。

### 验收判据

- [ ] `prompt=3 + 20 步 decode`,paged 路径与 Phase 2 连续 KV 路径的 **logits 逐步一致**,`check_close(rtol=1e-5)`(直接复用 `compare_stepwise_logits()`)
- [ ] `greedy` 采出的 token id 序列**完全相同**
- [ ] 块用量 == `ceil(23 / block_size)`,一块不多
- [ ] 序列结束后释放后 `num_free()` 回到初值
- [ ] 中途 `allocate()` 失败(把 `num_blocks` 调到刚好不够)→ 优雅返回 `false`,不越界写、不崩

---

## 5. Stage D – 对拍 + 边界

**目标:** 证明它真的对,以及在边界上不会烂。

### 必做的五组测试

1. **数值对拍** —— paged vs Phase 2 连续 cache,逐 token logits `rtol=1e-5`,这是 Stage D 的主判据。
2. **碎片化** —— 3 个序列长度不同(比如 5 / 11 / 3 个 token),**交错**分配和释放,验证:数据不串、`num_free()` 账目正确、最后全部归还。
3. **越界 / OOM** —— `ensure_capacity` 超过 `num_blocks`:返回 false,不写越界,allocator 状态未被破坏(后续正常分配仍能工作)。
4. **refcount 生命周期** —— `clone_shared()` 出一张表 → 析构原表 → 克隆表的数据仍然正确可读 → 析构克隆表 → 块归还。
5. **Benchmark** —— paged vs 连续 KV 的 20-token decode 耗时。**非 KV 侧也要加 warmup + median**(Phase 2 那个 benchmark 只给 KV 侧加了,数字系统性偏向 KV,别把这个毛病带过来)。

### 验收判据

- [ ] 五组全绿
- [ ] benchmark 数字记进 `benchmarks/`,并在 README 里写一句**为什么 paged 更慢** —— 说不清就是没真懂 gather 的代价
- [ ] fresh clone + `cmake` clean build 零 error(Phase 2 在这里栽过一次)

---

## 6. Ruthless-trim 砍点

时间不够时按这个顺序砍:

**可以砍:**
1. `MoveToEvictable` 策略的**真实实现**(接口和数据结构位置必须留,见留手 1)
2. `clone_shared()` 的**写时复制分裂** —cl— 只做 `incref` 就够了
3. 多序列 batching —— Stage D 第 2 组的交错测试已经覆盖了核心风险
4. Benchmark 的精细化(median/多轮),单次冷跑也能说明趋势

**不能砍:**
- `refcount` 本身
- `locate()` 的跨块正确性
- Stage D 第 1 组(数值对拍) —— 砍了这个整个 lab 就没有验收了
- 两处留手的**接口形状**

---

## 7. 打勾规则

`phase3-schedule.md` 里 Lab 1 的四个 Stage,**只有你亲口说"做完了"我才打 `[x]`**。cron 每天推的是进度卡,不代打、不越级。

`niobium-main` cron 周 — 08:00 PDT 恢复,第一张就是 Stage A 的进度卡。你这个周末先写也行,推的时候我按你实际进度调。