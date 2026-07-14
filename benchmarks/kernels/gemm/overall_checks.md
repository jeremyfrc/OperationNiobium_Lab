## Stage 0 – Naive（无 shared mem）

### checkpoints:

[X] CPU baseline 验证（1e-4 tolerance）
[X] 思考：A 的访问合并了吗？B 呢？
[X] ncu 跑一遍，记三数，理解为什么 memory-bound
[X] kernel: matmul_naive(float* A, float* B, float* C, int N)
[X] 每 thread 负责一个 C[row][col]，内层 for k 循环累加
[X] launch: 16x16 block，grid 覆盖整个 NxN 输出
[X] host 端：cudaMalloc / memcpy / correctness check vs CPU

验证：N=512 结果正确；nsys 看 global bandwidth（应远低于峰值）
---

在目前的写法下，A 的访问是广播 (Broadcast)，B 和 C 的访问是完美合并的 (Coalesced)
1. 矩阵 B 的访问 (B[k * N + col])
    - 在一个 Warp 执行同一条内层循环指令（k 相同）时，32 个线程同时请求 B[常量 + 0], B[常量 + 1]... B[常量 + 31]。
    - 物理内存地址是完美连续的！GPU 内存控制器会把这 32 个请求合并 (Coalesce) 成 1 扇区的连续内存读取操作。效率极高。

2. 矩阵 A 的访问 (A[row * K + k])
    - 在一个 Warp 内，row 是常量，k 也是常量。
    - 这意味着 32 个线程在同一时刻都在请求同一个内存地址！
    - 现代 GPU 的 L1 缓存极其聪明，它会读取一次这个数据，然后广播 (Broadcast) 给 Warp 内的所有线程。这也是一种非常高效的访问模式。
3. 矩阵 C 的写入 (C[row * N + col])
    - 同矩阵 B，col 是连续的，所以是完美的合并写入。
结论：Naive 版本的内存访问模式其实并不差，它符合硬件直觉。

用到的command:
`ncu -k matmul_naive -c 1 --set basic ./gemm_naive`

得到的三数是：
Compute (SM) Throughput  94.50%
Memory Throughput        94.50%
Achieved Occupancy       96.37%
divergence               100%
每条指令32条lane活跃 -> 32/32 = 100%
gld efficiency           2 sectors
理想合并 = 4， sectors/request，这里跑出来=2.

---
## Stage 1 – Tiled（shared memory）

### checkpoints:

[X] 两个 __syncthreads() 各在哪里？为什么都需要？
[X] As 和 Bs 的加载各自 coalesced 吗？（各自分析 warp 访问地址
[X] N 非 TILE_DIM 整数倍时会越界 → 加一行 if fix
[X] ncu 三数 vs Stage 0 对比：old efficiency 上来了多少？
[X] shared__ float As[T][T], Bs[T][T]; （T = TILE_WIDTH = 16）
[X] 外层 loop m = 0..N/T-1：每轮一个 tile
[X] 每 thread load：As[ty][tx] = A[row*N + m*T + tx]; Bs[ty][tx] = B[(m*T + ty)*N + Col];
[X] 内层 for k=0..T-1：Pvalue += As[ty][k] * Bs[k][tx];
[X] __syncthreads(); ← compute 结束后（防下一轮 load 踩脏 shmem）
[X] loop 结束后 C[row*N + Col] = Pvalue（只写一次）

得到的三数是：
Compute (SM) Throughput  95.18%
Memory Throughput        95.18%
Achieved Occupancy       96.44%
divergence               100%
bank conflict:           0.007% (4741/67,000,000)
32768 warps * 2048 (每个Warp跑64个tile,16个k*2次shared load)

---

验证：N=512 正确；对比 Stage 0 throughput，理论提升约 TILE_WIDTH 倍。

---

## Stage 2 – Optimized(register blocking / coarsening):写完 Stage 1 用数据驱动再上
**purpose**
1. register/thread tiling：每个线程算一个Micro-tile，一次shared load喂多次FMA -> 算数强度飙升 -> 访存管线不再是瓶颈。
2. double buffering/预取： 一边算当前tile，一边把下一个tile从global载进来，把访存延迟藏在计算后面；
3. 向量化访存(float4):一条指令搬128位，减少访存指令数
4. warp tiling、padding消除bank冲突(已经很低了)
```bash
ncu --metrics achieved_occupancy,\
    litex__t_sectors_pipe_lsu_mem_global_op_ld.avg.pct_of_peak_sustained_active,\
    smsp__thread_inst_executed_per_inst_executed.ratio \
    ./gemm_bench
```

分析：现在是 compute-bound 还是 memory-bound？
（roofline：测 achieved TFLOP/s，与 CGMA * bandwidth 比较）

[X] load A 越界补零：`As[ty][tx] = (Row < N && m*T+tx < N) ? A[...] : 0.f;`
[X] load B 同理
[X] 测试 N=1000
