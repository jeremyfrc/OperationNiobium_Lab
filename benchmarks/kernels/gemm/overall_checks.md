## Stage 0 – Naive（无 shared mem）

### checkpoints:

[ ] CPU baseline 验证（1e-4 tolerance）
[ ] 思考：A 的访问合并了吗？B 呢？
[ ] ncu 跑一遍，记三数，理解为什么 memory-bound
[ ] kernel: matmul_naive(float* A, float* B, float* C, int N)
[ ] 每 thread 负责一个 C[row][col]，内层 for k 循环累加
[ ] launch: 16x16 block，grid 覆盖整个 NxN 输出
[ ] host 端：cudaMalloc / memcpy / correctness check vs CPU

---

验证：N=512 结果正确；nsys 看 global bandwidth（应远低于峰值）

---
## Stage 1 – Tiled（shared memory）

### checkpoints:

[ ] 两个 __syncthreads() 各在哪里？为什么都需要？
[ ] As 和 Bs 的加载各自 coalesced 吗？（各自分析 warp 访问地址）
[ ] N 非 TILE_DIM 整数倍时会越界 → 加一行 if fix
[ ] ncu 三数 vs Stage 0 对比：old efficiency 上来了多少？
[ ] shared__ float As[T][T], Bs[T][T]; （T = TILE_WIDTH = 16）
[ ] 外层 loop m = 0..N/T-1：每轮一个 tile
[ ] 每 thread load：As[ty][tx] = A[row*N + m*T + tx]; Bs[ty][tx] = B[(m*T + ty)*N + Col];
[ ] 内层 for k=0..T-1：Pvalue += As[ty][k] * Bs[k][tx];
[ ] __syncthreads(); ← compute 结束后（防下一轮 load 踩脏 shmem）
[ ] loop 结束后 C[row*N + Col] = Pvalue（只写一次）

---

验证：N=512 正确；对比 Stage 0 throughput，理论提升约 TILE_WIDTH 倍。

---

## Stage 2 – Optimized(register blocking / coarsening):写完 Stage 1 用数据驱动再上

```bash
ncu --metrics achieved_occupancy,\
    litex__t_sectors_pipe_lsu_mem_global_op_ld.avg.pct_of_peak_sustained_active,\
    smsp__thread_inst_executed_per_inst_executed.ratio \
    ./gemm_bench
```

分析：现在是 compute-bound 还是 memory-bound？
（roofline：测 achieved TFLOP/s，与 CGMA * bandwidth 比较）

[ ] load A 越界补零：`As[ty][tx] = (Row < N && m*T+tx < N) ? A[...] : 0.f;`
[ ] load B 同理
[ ] 测试 N=1000
