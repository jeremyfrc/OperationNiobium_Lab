================================================================
 Lab1 — Paged KV Attention
 Full test suite + benchmark record
================================================================
Date   : <Sep 28 2026>
Build  : Release (-O2 -DNDEBUG)
Host   : <Big_brother/WSL2 Ubuntu x86_64>
Config : tiny (d_model, n_layers, n_heads, n_kv_heads, d_head)
Workload: prompt=3, decode new_tokens=20 ; warmup + median over 20 runs

================================================================
 ctest -V  (14 tests)
================================================================

$ ctest --test-dir build -V
Internal ctest changing into directory: /home/frc28/OperationNiobium_Lab/kernels/build
UpdateCTestConfiguration  from :/home/frc28/OperationNiobium_Lab/kernels/build/DartConfiguration.tcl
UpdateCTestConfiguration  from :/home/frc28/OperationNiobium_Lab/kernels/build/DartConfiguration.tcl
Test project /home/frc28/OperationNiobium_Lab/kernels/build
Constructing a list of tests
Done constructing a list of tests
Updating test list for fixtures
Added 0 tests to meet fixture requirements
Checking test dependency graph...
Checking test dependency graph end
test 1
      Start  1: test_block_allocator

1: Test command: /home/frc28/OperationNiobium_Lab/kernels/build/paged_kv/test_block_allocator
1: Working Directory: /home/frc28/OperationNiobium_Lab/kernels/build/paged_kv
1: Test timeout computed to be: 10000000
1: test_block_allocator: ALL PASSED
 1/14 Test  #1: test_block_allocator .............   Passed    0.00 sec
test 2
      Start  2: test_allocate_table

2: Test command: /home/frc28/OperationNiobium_Lab/kernels/build/paged_kv/test_allocate_table
2: Working Directory: /home/frc28/OperationNiobium_Lab/kernels/build/paged_kv
2: Test timeout computed to be: 10000000
2: test_allocate_table: ALL PASSED
 2/14 Test  #2: test_allocate_table ..............   Passed    0.00 sec
test 3
      Start  3: test_paged_kv_cache

3: Test command: /home/frc28/OperationNiobium_Lab/kernels/build/paged_kv/test_paged_kv_cache
3: Working Directory: /home/frc28/OperationNiobium_Lab/kernels/build/paged_kv
3: Test timeout computed to be: 10000000
3: test_paged_kv_cache: ALL PASSED
 3/14 Test  #3: test_paged_kv_cache ..............   Passed    0.00 sec
test 4
      Start  4: test_sequence_manager

4: Test command: /home/frc28/OperationNiobium_Lab/kernels/build/paged_kv/test_sequence_manager
4: Working Directory: /home/frc28/OperationNiobium_Lab/kernels/build/paged_kv
4: Test timeout computed to be: 10000000
4: test_sequence_manager: ALL PASSED
 4/14 Test  #4: test_sequence_manager ............   Passed    0.00 sec
test 5
      Start  5: test_rmsnorm

5: Test command: /home/frc28/OperationNiobium_Lab/kernels/build/models/transformer/test_rmsnorm
5: Working Directory: /home/frc28/OperationNiobium_Lab/kernels/models/transformer
5: Test timeout computed to be: 10000000
5: ========================================
5:        Running RMSNorm Unit Test        
5: ========================================
5: [Test] Running RMSNorm Op Test... PASSED! ✅
5: 
5: All local test cases passed successfully! 🚀
5: ✅ [PASS] RMSNorm test passed!
5: 
5: All python ref RMSNorm test cases passed successfully! 🚀
 5/14 Test  #5: test_rmsnorm .....................   Passed    0.00 sec
test 6
      Start  6: test_rope

6: Test command: /home/frc28/OperationNiobium_Lab/kernels/build/models/transformer/test_rope
6: Working Directory: /home/frc28/OperationNiobium_Lab/kernels/models/transformer
6: Test timeout computed to be: 10000000
6: ========================================
6:         Running RoPE Unit Test          
6: ========================================
6: [Test] Running In-place RoPE Op Test... PASSED! ✅
6: 
6: All test cases passed successfully! 🚀
6: ✅ [PASS] RoPE test passed!
6: 
6: All python ref RoPE test cases passed successfully! 🚀
 6/14 Test  #6: test_rope ........................   Passed    0.00 sec
test 7
      Start  7: test_softmax

7: Test command: /home/frc28/OperationNiobium_Lab/kernels/build/models/transformer/test_softmax
7: Working Directory: /home/frc28/OperationNiobium_Lab/kernels/models/transformer
7: Test timeout computed to be: 10000000
7: ========================================
7:    Running Transformer C++ Unit Tests   
7: ========================================
7: [Test] Running Softmax Op Test... PASSED! ✅
7: [Test] Running Matmul Op Test... PASSED! ✅
7: 
7: All active unit tests passed successfully! 🚀
7: Loaded x_data size: 512
7: Loaded ref_out size: 512
7: ✅ [PASS] Softmax test passed!
7: 
7: All python ref Softmax test cases passed successfully! 🚀
 7/14 Test  #7: test_softmax .....................   Passed    0.00 sec
test 8
      Start  8: test_swiglu

8: Test command: /home/frc28/OperationNiobium_Lab/kernels/build/models/transformer/test_swiglu
8: Working Directory: /home/frc28/OperationNiobium_Lab/kernels/models/transformer
8: Test timeout computed to be: 10000000
8: ========================================
8:        Running SwiGLU Unit Test         
8: ========================================
8: [Test] Running SwiGLU Forward FFN Test... PASSED! ✅
8: 
8: All local test cases passed successfully! 🚀
8: ✅ [PASS] SwiGLU test passed!
8: 
8: All python ref SwiGLU test cases passed successfully! 🚀
 8/14 Test  #8: test_swiglu ......................   Passed    0.00 sec
test 9
      Start  9: test_attention_block

9: Test command: /home/frc28/OperationNiobium_Lab/kernels/build/models/transformer/test_attention_block
9: Working Directory: /home/frc28/OperationNiobium_Lab/kernels/models/transformer
9: Test timeout computed to be: 10000000
9: ========================================
9:        Running Attention Block Test     
9: ========================================
9: ✅ [PASS] Attention block test passed!
 9/14 Test  #9: test_attention_block .............   Passed    0.00 sec
test 10
      Start 10: test_full_forward

10: Test command: /home/frc28/OperationNiobium_Lab/kernels/build/models/transformer/test_full_forward
10: Working Directory: /home/frc28/OperationNiobium_Lab/kernels/models/transformer
10: Test timeout computed to be: 10000000
10: ✅ Stage 3: Full Forward Test Passed!
10/14 Test #10: test_full_forward ................   Passed    0.00 sec
test 11
      Start 11: test_generate

11: Test command: /home/frc28/OperationNiobium_Lab/kernels/build/models/transformer/test_generate
11: Working Directory: /home/frc28/OperationNiobium_Lab/kernels/models/transformer
11: Test timeout computed to be: 10000000
11: Prompt: 1 5 42 
11: Generated: 1 5 42 5 41 22 38 22 
11: ✅ Generation Test Success!
11/14 Test #11: test_generate ....................   Passed    0.00 sec
test 12
      Start 12: test_generate_kv

12: Test command: /home/frc28/OperationNiobium_Lab/kernels/build/models/transformer/test_generate_kv
12: Working Directory: /home/frc28/OperationNiobium_Lab/kernels/models/transformer
12: Test timeout computed to be: 10000000
12: === KV-Cache Generate Test ===
12: Prompt: 1 5 42 
12: KV   generated: 1 5 42 5 41 22 38 22 
12: NonKV generated: 1 5 42 5 41 22 38 22 
12: ✅ KV logits match non-KV logits (rtol=1e-5)
12: ✅ KV generate matches non-KV generate (tokens + logits)!
12/14 Test #12: test_generate_kv .................   Passed    0.00 sec
test 13
      Start 13: benchmark

13: Test command: /home/frc28/OperationNiobium_Lab/kernels/build/models/transformer/benchmark
13: Working Directory: /home/frc28/OperationNiobium_Lab/kernels/models/transformer
13: Test timeout computed to be: 10000000
13: ===== Benchmark (new_tokens=20) =====
13: Non-KV : 0.841083 ms  (0.0420541 ms/token, 23778.9 tok/s)
13: KV     : 0.095898 ms  (0.0047949 ms/token, 208555 tok/s)
13: Paged  : 0.101935 ms  (0.00509675 ms/token, 196203 tok/s)
13: Speedup (Non-KV / KV)    : 8.7706x
13: Speedup (Non-KV / Paged) : 8.25117x
13: ✅ outputs identical (Non-KV == KV == Paged)
13/14 Test #13: benchmark ........................   Passed    0.02 sec
test 14
      Start 14: test_paged_vs_contiguous

14: Test command: /home/frc28/OperationNiobium_Lab/kernels/build/models/transformer/test_paged_vs_contiguous
14: Working Directory: /home/frc28/OperationNiobium_Lab/kernels/models/transformer
14: Test timeout computed to be: 10000000
14: === Stage D: paged vs contiguous (prompt=3 + 20 decode) ===
14: greedy seq: 1 5 42 5 41 22 38 22 23 45 28 27 27 27 22 46 5 45 22 38 33 46 5 17 
14: [bad pos_offset] caught: NB_CHECK failed: forward_paged(): table token number and position offset mismatch! | pos_offset == table.num_tokens()
14: test_paged_vs_contiguous: ALL PASSED
14/14 Test #14: test_paged_vs_contiguous .........   Passed    0.00 sec

100% tests passed, 0 tests failed out of 14

Total Test time (real) =   0.05 sec

================================================================
 BENCHMARK ANALYSIS — why is paged slower than contiguous KV?
================================================================

Result (from benchmark above):
  Non-KV : 0.841083 ms  (0.0420541 ms/token,  23778.9 tok/s)
  KV     : 0.095898 ms  (0.0047949 ms/token, 208555   tok/s)
  Paged  : 0.101935 ms  (0.00509675 ms/token,196203   tok/s)
  -> Paged is ~6.3% slower than contiguous KV (0.1019 vs 0.0959 ms).

Mechanism:
  - Contiguous KV: decode step t reads KV[0, t) directly via a
    contiguous offset. No copy. Per step O(t) reads, and the data
    is already laid out the way attention wants it.
  - Paged KV: KV for a sequence is scattered across physical blocks
    in arbitrary order. Before attention, every decode step calls
    gather() to copy KV[0, t) out of the blocks into a contiguous
    scratch buffer. That is an extra O(t) memcpy each step, so
    O(T^2) total copying over T decode steps -- on top of the
    attention math itself.

Why only ~6% here:
  - The tiny config (few heads / small d_head) and short context
    (20 tokens) make the copied volume small relative to the
    matmul cost, so gather overhead is minor.

Why it matters / Lab2:
  - As context length and KV size grow, the O(T^2) gather copy
    dominates and paged attention would be substantially slower
    than the contiguous baseline.
  - Lab2's goal is to eliminate the materialized gather: run
    attention directly over the blocks (block-wise / online-softmax
    attention) so no full-history copy is ever needed.

================================================================
 BENCHMARK AFTER FIX
================================================================
13: Test command: /home/frc28/OperationNiobium_Lab/kernels/build/models/transformer/benchmark
13: Working Directory: /home/frc28/OperationNiobium_Lab/kernels/models/transformer
13: Test timeout computed to be: 10000000
13: ===== Length sweep (prompt=3) =====
13: new_tokens |  KV ms (IQR)      |  Paged ms (IQR)   | Paged-KV | ratio
13: -----------+-------------------+-------------------+----------+------
13: 20         | 0.096807 (0.004726) | 0.100073 (0.001329) | 0.003266 | 1.03374x
13:   ✅ identical
13: 60         | 0.44094 (0.011689) | 0.454759 (0.0098) | 0.013819 | 1.03134x
13:   ✅ identical
13: 120         | 1.43464 (0.021906) | 1.4737 (0.019455) | 0.039061 | 1.02723x
13:   ✅ identical
13: 250         | 5.20928 (0.023845) | 5.37565 (0.073482) | 0.166374 | 1.03194x
13:   ✅ identical
13/14 Test #13: benchmark ........................   Passed    0.31 sec

================================================================
 Length sweep (prompt=3, env OUTSIDE timer, runs=100)
================================================================
 new_tokens | KV  min/med/IQR (us)   | Paged min/med/IQR (us) | d-min | d-med
 -----------+-----------------------+------------------------+-------+------
 20         |  94.31/ 95.60/ 2.29    |  96.76/ 98.41/ 5.31    |  2.45 |  2.82
 60         | 434.25/440.81/11.91    | 445.61/452.63/10.84    | 11.36 | 11.82
 120        |1409.20/1429.45/16.01   |1443.21/1463.15/13.69   | 34.01 | 33.70
 250        |5171.68/5219.34/64.53   |5335.21/5390.17/70.58   |163.53 |170.83

 Method : each config, warmup then 100 timed runs; report min, median,
          IQR. min and median agree closely -> the gap is real, not jitter.
          allocator/table/cache built ONCE outside the timer (both paths);
          paged table is reset (not timed) between runs.

================================================================
 NEW Analysis
================================================================
1) The paged penalty is the materialized gather, NOT malloc.
   Earlier a 6.3% gap looked like gather cost; it was almost entirely
   per-layer allocations of k_buf/v_buf inside attention_paged_forward
   (84 allocs + zero-init per generate). After hoisting these into a
   reusable scratch buffer in PagedKVCache, and aligning the timer,
   the residual gap is the real gather.

2) The gap grows super-linearly with context.
   d-min: 2.45 -> 11.36 -> 34.01 -> 163.53 us  (20 -> 60 -> 120 -> 250).
   length x12.5 (20->250) but gap x67  ->  consistent with the O(T^2)
   gather copy (every decode step copies [0,t) KV out of the blocks).
   Relative cost: ~2.6% (20) to ~3.2% (250).

3) Why it matters / Lab 2 target.
   The cost is a materialized gather over the full history. At this tiny
   scale it is only a few percent, but the absolute cost grows O(T^2),
   so at longer contexts it dominates. Lab 2 is about eliminating the
   materialized gather: attend directly over blocks (block-wise /
   online-softmax), so no full-history copy is ever materialized.

================================================================
 Caveats
================================================================
- 20-token point is near the noise floor: d-min=2.45us vs IQR 2.3/5.3us.
  Signal is clean from 60 tokens up (gap >> IQR).
- Build is Release (-O2 -DNDEBUG). With -DNDEBUG the assert_invariant()
  checks in the hot path are compiled out. (Hardcoded -DNDEBUG is a
  known flag-hygiene issue: should come from CMAKE_BUILD_TYPE instead;
  to be fixed next lab.)
- Numbers are host- and build-specific; re-measure on target hardware.

========================================================================
REVISED ANALYSIS (supersedes points 1-3 of "NEW Analysis")
========================================================================
Three corrections after re-deriving the numbers. R2 is the one thatturns the O(T^2) claim from an assertion into a measurement; R3reverses the Lab 2 framing.

R1.
Malloc was about HALF the original gap, not almost all of it.    
before fix : 101.935 - 95.898 = 6.04 us    
after fix : 98.410 - 95.600 = 2.82 us (d-min 2.45 us)    
The KV column barely moved (95.90 -> 95.60), so aligning the timer    
changed almost nothing and the two runs are comparable. Hoisting    
k_buf/v_buf therefore recovered ~3.2 us of 6.0 us, i.e. ~53%.    
The remaining ~47% is the gather itself. Point 1 above claims the    
penalty is "NOT malloc"; the correct statement is that malloc and    
gather were roughly half each, and only gather is left.

R2.
The gap IS the gather copy. The evidence is effective bandwidth,    
not the growth exponent.    
A pure T^2 law predicts 12.5x length -> 156x gap. Observed is 67x,    
so the fitted exponent is log(67)/log(12.5) = 1.66, not 2. Segment    
by segment it is still climbing: 1.40 (20->60), 1.58 (60->120),    
2.14 (120->250). Quoting "x67, consistent with O(T^2)" overstates    
a 2.3x miss.

Dividing the gap by the theoretical copy volume resolves it. Each    
decode step at position t gathers 2*t*n_kv_heads*d_head floats    
(K and V) per layer = 128t bytes over 2 layers, so the whole run    
copies sum(128t) ≈ 64*T^2 bytes.

T copy volume gap (d-min) effective bandwidth      
20 25.6 KB 2.45 us 10.4 GB/s      
60 230 KB 11.36 us 20.3 GB/s      
120 922 KB 34.01 us 27.1 GB/s      
250 4.0 MB 163.53 us 24.5 GB/s

Bandwidth rises from 10 GB/s and saturates near 25 GB/s: a textbook
memcpy curve, per-call overhead dominating small copies and L2/L3
bandwidth capping large ones. So the sub-quadratic exponent is not
a counter-example, it is the confirmation. The gap is exactly that
copy; it only looks sub-quadratic because memcpy gets more
efficient as the copy grows.

R3.
The RELATIVE cost is flat, not growing. Point 3 is contradicted by
this table's own numbers.
paged overhead: 2.60% (20) 2.62% (60) 2.41% (120) 3.16% (250)
It does not trend up, because the contiguous baseline is itself
~O(T^2) here: its growth exponent is 1.39, 1.70, 1.77 over the same
three intervals. Numerator and denominator are the same order, so
the ratio barely moves.

Lab 2 therefore cannot be motivated by "gather eventually dominates"
-- this data says it does not, at least not on this shape. The
honest motivation:

Paging costs a steady ~3% that buys no arithmetic at all. It is a
full-history copy materialized once per layer per decode step,
purely to hand attention a contiguous buffer. The real price is
memory bandwidth and, under batching, bandwidth contention -- not
an asymptotic blow-up. Lab 2 removes the materialization: attend
directly over the blocks (block-wise / online-softmax) so the
copy never happens.

Getting this right matters because the asymptotic version of the
claim is falsifiable and already falsified above; the bandwidth
version is what the measurement actually supports.