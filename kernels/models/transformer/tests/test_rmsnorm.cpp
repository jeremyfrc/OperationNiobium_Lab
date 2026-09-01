#include <iostream>
#include <cmath>
#include <cassert>
#include "tensor.h"
#include "rmsnorm.h" // 引用独立的 RMSNorm 算子头文件
#include "test_utils.h" // 引用测试工具头文件

static bool approx_equal(float a, float b, float epsilon = 1e-4f) {
    return std::fabs(a - b) < epsilon;
}

void test_rmsnorm_op_local() {
    std::cout << "[Test] Running RMSNorm Op Test... " << std::flush;

    // 输入 1 行 4 列: x = [1.0, 2.0, 3.0, 4.0]
    // 平方和 sum_sq = 1 + 4 + 9 + 16 = 30.0
    // 均方 mean_sq = 30.0 / 4 = 7.5
    // RMS = sqrt(7.5 + 1e-5) ≈ 2.7386127
    Tensor x({1, 4});
    x.at({0, 0}) = 1.0f; x.at({0, 1}) = 2.0f; x.at({0, 2}) = 3.0f; x.at({0, 3}) = 4.0f;

    Tensor weight({4});
    weight.at({0}) = 1.0f; weight.at({1}) = 1.0f; weight.at({2}) = 1.0f; weight.at({3}) = 1.0f;

    Tensor out({1, 4});

    rmsnorm(x, weight, out, 1e-5f);

    float expected_rms = std::sqrt(7.5f + 1e-5f);
    assert(approx_equal(out.at({0, 0}), 1.0f / expected_rms));
    assert(approx_equal(out.at({0, 1}), 2.0f / expected_rms));
    assert(approx_equal(out.at({0, 2}), 3.0f / expected_rms));
    assert(approx_equal(out.at({0, 3}), 4.0f / expected_rms));

    std::cout << "PASSED! ✅\n";
}

int test_rms_ref() {
    try{
        auto x_data = loadBinFile("tests/ref/data/rmsnorm_input.bin");
        auto w_data = loadBinFile("tests/ref/data/rmsnorm_weight.bin");
        auto ref_out = loadBinFile("tests/ref/data/rmsnorm_out.bin");

        Tensor input({2, 16, 64}, x_data);
        Tensor weight({64}, w_data);
        Tensor output({2, 16, 64});

        rmsnorm(input, weight, output, 1e-5f);

        bool ok = check_close(output.data(), ref_out.data(), ref_out.size(), 1e-4f, 1e-5f);
        if (ok) {
            std::cout << "✅ [PASS] RMSNorm test passed!" << std::endl;
            return 0; // 0 表示测试成功
        } else {
            std::cout << "❌ [FAIL] RMSNorm test failed!" << std::endl;
            return 1; // 1 表示测试失败
        }
    } catch (const std::exception& e){
        std::cerr << "Error: " << e.what() << std::endl;
        return 1; // 1 表示测试失败
    }
}

int main() {
    std::cout << "========================================\n";
    std::cout << "       Running RMSNorm Unit Test        \n";
    std::cout << "========================================\n";

    test_rmsnorm_op_local();
    std::cout << "\nAll local test cases passed successfully! 🚀\n";

    int s = test_rms_ref();
    if (s != 0) {
        std::cerr << "RMSNorm reference test failed!" << std::endl;
        return s; // 返回错误码
    }
    std::cout << "\nAll python ref test cases passed successfully! 🚀\n";

    return 0;
}