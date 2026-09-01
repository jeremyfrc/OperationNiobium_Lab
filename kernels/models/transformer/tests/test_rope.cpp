#include <iostream>
#include <cmath>
#include <cassert>
#include "tensor.h"
#include "rope.h"

static bool approx_equal(float a, float b, float epsilon = 1e-4f) {
    return std::fabs(a - b) < epsilon;
}

void test_rope_inplace_op() {
    std::cout << "[Test] Running In-place RoPE Op Test... " << std::flush;

    // 测试 1：pos = 0 处，旋转后应该保持原值
    Tensor x_pos0({1, 1, 2});
    x_pos0.at({0, 0, 0}) = 1.0f;
    x_pos0.at({0, 0, 1}) = 2.0f;

    rope_inplace(x_pos0, 1, /*pos_offset=*/0);

    assert(approx_equal(x_pos0.at({0, 0, 0}), 1.0f));
    assert(approx_equal(x_pos0.at({0, 0, 1}), 2.0f));

    // 测试 2：pos = 1, head_dim = 2, 原地修改 [1.0, 0.0]
    // 旋转后应变成 [cos(1.0), sin(1.0)]
    Tensor x_pos1({1, 1, 2});
    x_pos1.at({0, 0, 0}) = 1.0f;
    x_pos1.at({0, 0, 1}) = 0.0f;

    rope_inplace(x_pos1, 1, /*pos_offset=*/1);

    assert(approx_equal(x_pos1.at({0, 0, 0}), std::cos(1.0f)));
    assert(approx_equal(x_pos1.at({0, 0, 1}), std::sin(1.0f)));

    std::cout << "PASSED! ✅\n";
}

int main() {
    std::cout << "========================================\n";
    std::cout << "        Running RoPE Unit Test          \n";
    std::cout << "========================================\n";

    test_rope_inplace_op();

    std::cout << "\nAll test cases passed successfully! 🚀\n";
    return 0;
}