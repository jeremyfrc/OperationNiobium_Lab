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