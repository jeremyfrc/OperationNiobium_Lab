# GEMM 手写脚手架（三阶段）

自己写！每阶段用 `ncu` 测完把三数写进 `kernels/gemm/README`，然后拉 git 到主 space 让我 review。  
三数指标：
- `achieved_occupancy`
- `gld efficiency`（全局内存加载合并率）
- `warp execution efficiency`（divergence）

---

## Stage 0 – Naive（无 shared mem）

直接写（每线程一个输出，全走 global）——不用配置，写就完了，拿它当基线。

```cpp
// kernels/gemm/naive.cu
global__void gemm_naive(float* A, float* B, float* C, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.x * blockDim.x + threadIdx.x; // 注意：row 和 col 都用了 blockIdx.x，应改为 blockIdx.y？
    // TODO: 边界检查 + k 循环累加内积
    // A[row*N+k], B[k*N+col]
}

// launch: dim3(N/16, N/16), dim3(16, 16)
```
检查点：

[ ] CPU baseline 验证（1e-4 tolerance）
[ ] 思考：A 的访问合并了吗？B 呢？
[ ] ncu 跑一遍，记三数，理解为什么 memory-bound
[ ] kernel: matmul_naive(float* A, float* B, float* C, int N)
[ ] 每 thread 负责一个 C[row][col]，内层 for k 循环累加
[ ] launch: 16x16 block，grid 覆盖整个 NxN 输出
[ ] host 端：cudaMalloc / memcpy / correctness check vs CPU

验证：N=512 结果正确；nsys 看 global bandwidth（应远低于峰值）

---

## Stage 1 – Tiled（shared memory）
tiled + shared-mem：这一步才需要 Ch6 的 tile 节奏垫底 → 写之前把那段想清楚。

```cpp
#define TILE_DIM 16
__global__ void gemm_tiled(float* A, float* B, float* C, int N) {
    shared__ float As[TILE_DIM][TILE_DIM];
    shared__ float Bs[TILE_DIM][TILE_DIM];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.x * TILE_DIM + ty;
    int col = blockIdx.x * TILE_DIM + tx; // 同样注意 blockIdx.x 应改为 blockIdx.y？
    float sum = 0.0f;
    for (int t = 0; t < N / TILE_DIM; t++) {
        // TODO: As[ty][tx] = A[row][t*TILE_DIM + tx] ← 合并？
        // TODO: Bs[ty][tx] = B[t*TILE_DIM + ty][col] ← 合并？
        // syncthreads();
        // TODO: 局部内积 k=0..TILE_DIM-1
        // syncthreads();
    }
    // TODO: C[row*N+col] = sum
}
```

检查点：

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

验证：N=512 正确；对比 Stage 0 throughput，理论提升约 TILE_WIDTH 倍。


---

## Stage 2 – Optimized(register blocking / coarsening):写完 Stage 1 用数据驱动再上。(stretch goal， will check when get time)

`ncu`实测命令bash：

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