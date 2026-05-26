#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>


void sqrtSerial(int N,
                float initialGuess,
                float values[],
                float output[])
{

    static const float kThreshold = 0.00001f;

    for (int i=0; i<N; i++) {

        float x = values[i];
        float guess = initialGuess;

        float error = fabs(guess * guess * x - 1.f);

        while (error > kThreshold) {
            guess = (3.f * guess - x * guess * guess * guess) * 0.5f;
            error = fabs(guess * guess * x - 1.f);
        }

        output[i] = x * guess;
    }
}

void sqrtSimd256(int N, float initialGuess, float values[], float output[]) 
{

  __m256 kThreshold_v = _mm256_set1_ps(0.00001f); //AVX的256bit寄存器，装了8个float , kthreshold 用于判断 Newton 是否收敛
  __m256 three_v = _mm256_set1_ps(3.0f); // 常量 3
  __m256 one_v = _mm256_set1_ps(1.0f); // 常量 1
  __m256 half_v = _mm256_set1_ps(0.5f); // 常量 0.5
  __m256 zero_v = _mm256_setzero_ps(); // 常量 0
  __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)); // sign_mask: IEEE754 float, 清除 sign bit，保留其他位

  for (int i = 0; i < N; i += 8) {
    // 1. 使用 loadu 避免对齐导致的 Segfault
    __m256 x_v = _mm256_loadu_ps(values + i);  // u = unaligned，不是32-byte对齐的; _mm256_load_ps 若地址未对齐可能直接SIGSEGV
    __m256 guess_v = _mm256_set1_ps(initialGuess); 

    // 2. 预处理：如果是 0，直接标记，防止死循环
    __m256 is_zero_mask = _mm256_cmp_ps(x_v, zero_v, _CMP_EQ_OQ); //检测 x == 0

    // 初次计算 error:
    // guess = 1 / sqrt(x) --> x * guess ^ 2 = 1
    // error = abs(x * guess^2 - 1)
    __m256 term1_v = _mm256_mul_ps(_mm256_mul_ps(guess_v, guess_v), x_v);
    __m256 error_v = _mm256_and_ps(_mm256_sub_ps(term1_v, one_v), sign_mask);
      
    // 关键：mask 必须排除掉那些输入为 0 的通道
    // active_mask为未收敛的Lane 即 error > kthreshold
    __m256 active_mask = _mm256_cmp_ps(error_v, kThreshold_v, _CMP_GT_OQ);
    active_mask = _mm256_andnot_ps(is_zero_mask, active_mask); // anddot(a,b) = (~a) & b

    int iterCount = 0;
    // 防止active_mask = 0,那样error 会一直等于1
    while (_mm256_movemask_ps(active_mask) && iterCount < 100) 
    {
      // Newton-Raphson 迭代
      // 计算新的guess：yn+1 = 0.5 * (3 * yn - x * yn ^ 3)
      __m256 term3_v = _mm256_mul_ps(_mm256_mul_ps(x_v, guess_v), _mm256_mul_ps(guess_v, guess_v));
      guess_v = _mm256_mul_ps(_mm256_sub_ps(_mm256_mul_ps(three_v, guess_v), term3_v), half_v);

      // 更新 error 和 mask
      term1_v = _mm256_mul_ps(_mm256_mul_ps(guess_v, guess_v), x_v);
      error_v = _mm256_and_ps(_mm256_sub_ps(term1_v, one_v), sign_mask);
      active_mask = _mm256_cmp_ps(error_v, kThreshold_v, _CMP_GT_OQ);
      active_mask = _mm256_andnot_ps(is_zero_mask, active_mask);
      iterCount++;
    }

    __m256 final_output_v = _mm256_mul_ps(x_v, guess_v);
    // 如果输入是 0，结果强制设为 0
    final_output_v = _mm256_andnot_ps(is_zero_mask, final_output_v);
    
    // 3. 使用 storeu 存回, 不要求对齐
    _mm256_storeu_ps(output + i, final_output_v);
  }
}