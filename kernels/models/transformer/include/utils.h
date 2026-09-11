#pragma once
#include <iostream>

// 返回 logits 中最大值的下标
inline int argmax(const float* logits, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i)
        if (logits[i] > logits[best]) best = i;
    return best;
}
