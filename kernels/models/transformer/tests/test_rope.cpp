#include <iostream>
#include <cmath>
#include <cassert>
#include "tensor.h"
#include "rope.h"
#include "test_utils.h" // 引用测试工具头文件

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

int test_rope_ref() {
    try {
        std::string base = "tests/ref/data/";

        // 对应 Dump 时的 Shape: [1, 8, 2, 16]
        auto q_data = loadBinFile(base + "rope_input.bin");
        auto ref_out = loadBinFile(base + "rope_out.bin");

        Tensor q({1, 8, 2, 16}, q_data);
        Tensor actual_out({1, 8, 2, 16});

        // 调用 rope: numHeads = 2, posOffset = 0, base = 10000.0f
        rope(q, actual_out, 2, 0, 10000.0f);

        bool ok = check_close(actual_out.data(), ref_out.data(), ref_out.size(), 1e-4f, 1e-5f);
        if (ok) {
            std::cout << "✅ [PASS] RoPE test passed!" << std::endl;
            return 0;
        } else {
            std::cout << "❌ [FAIL] RoPE test passed!" << std::endl;
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error in RoPE test: " << e.what() << std::endl;
        return 1;
    }
}

int main() {
    std::cout << "========================================\n";
    std::cout << "        Running RoPE Unit Test          \n";
    std::cout << "========================================\n";

    test_rope_inplace_op();
    std::cout << "\nAll test cases passed successfully! 🚀\n";

    int s = test_rope_ref();
    if (s != 0) {
        std::cerr << "RoPE reference test failed!" << std::endl;
        return s; // 返回错误码
    }
    std::cout << "\nAll python ref RoPE test cases passed successfully! 🚀\n";
    return 0;
}