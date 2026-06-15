# Scan 吃透笔记

## 0. 心智模型(理解所有索引的钥匙)
把数组当成一棵二叉树，每个节点的和存在它覆盖区间的最右一格
- stride = 当自取件的办款(up-sweep逐步x2, down-sweep逐步➗️2)
- 每个线程管2个元素--> 一个Block处理'Elements_per_block' = 'threads_per_block'*2 = 512个
- exclusive scan: out[i] = in[0] + ... + in[i-1], out[0] = 0。

索引公式
```
ai = stride * (2 * tid + 1) - 1 //左半子区间的最右格
bi = stride * (2 * tid + 2) - 1 //右半子区间的最右格 （父节点存和处）
```
为什么`-1`：数组0-indexed，区间长度'2*stride'，某段最右格 = 段数 x 长度 - 1。


---
## 1. 'local_block_scan' -- 心脏：单块exclusive scan

五步：
**step1 加载：** 's_data[2tid] = device_data[block_offfset + 2tid]'(越界补0)，'s_data[2tid+1]’同理--> ‘syncthreads()'。

**step2 up-sweep(规约/reduce)：** 'for d= THREADS_PER_BLOCK-->1(>>1)'，'stride 1->2->4...'
```cpp
if (tid < d) {s_data[bi] += s_data[ai];}
stride *= 2; __syncthreads();
```
'tid < d' 让活跃线程每轮减半，跑完末格 = 全块总和。

**step3 取决总和 + 清根：**
```
if (tid == 0) { block_sum[block] = s_data[EPB-1]; s_data[EPB-1] = 0;}
```
**清根是exclusive的命门**：把根设为identity(0), down-sweep 才会灌前缀和而非含自己的和。

**step4 Down-sweep：** 'for d=1->TPB(<<1)`,'stride`反向➗️2
```
if(tid < d) { int t = s_data[ai]; s_data[ai] = s_data[bi]; s_data[bi] += t; }
```
含义：左孩子拿父值，右孩子 = 父+旧左值。

**step5 写回：** global memory
example: input [3,1,7,0,4,1,6,3]
up-sweep之后:  [3,4,7,11,4,5,6,25]
down-sweep之后: [0,3,4,11,11,15,16,22]


---
## 2. 'accumulate_block_offsets' -- 跨块补偏移
单块只在块内正确，块之间还差前面所有块的总和。block_sum经过递归scan之后，block_sum[i] = 第i块之前所有块的总和 = 该块要加的偏移。
```
if (blockIdx.x == 0) return; // 第0块偏移=0
device_data[idx] += block_sums[blockIdx.x];
```

## 3. 'exclusive_scvan_recursive' -- 引擎(分治三段式)
```
base: length<=1 return 
1. local_block_scan   //每块各自exclusive scan + 吐出block_sums
2. if numBlocks>1:
    exclusive_scan_recursive(block_nums, numBlocks)   //递归：对总快总和再scan
    accumulate_block_offsets                          // 偏移加回每块
```

**为什么必须递归/多次kernel launch：** `__syncthreads`只通不用块内；跨块协调的唯一全局barrier = kernel边界。
block_nums自己也可能超过一个块 --> 同样问题 --> 递归。
**一句话总结：** 1) 局部 --> 汇总递归 3) --> 回填 2)。 find_repeats的flag -> scan -> scatter同壳








