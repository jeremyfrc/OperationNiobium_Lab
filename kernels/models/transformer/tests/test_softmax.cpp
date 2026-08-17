#include <iostream>
#include <cmath>
#include <cassert>
#include "tensor.h"
#include "ops.h"

static bool approx_equal(float a, float b, float epsilon = 1e-4f){
    return std::fabs(a - b) < epsilon;
}

void test_softmax_op(){
    std::cout << "[Test] Running Softmax Op Test... " << std::flush;

    Tensor x({2, 3});
    x.at({0, 0}) = 1.0f;
    x.at({0, 1}) = 2.0f;
    x.at({0, 2}) = 3.0f;
    x.at({1, 0}) = 0.0f;
    x.at({1, 1}) = 0.0f;
    x.at({1, 2}) = 0.0f;

    softmax_inplace(x);

    // 第 0 行概率和为 1.0，且递增
    float sum0 = x.at({0, 0}) + x.at({0, 1}) + x.at({0, 2});
    assert(approx_equal(sum0, 1.0f));
    assert(x.at({0, 2}) > x.at({0, 1}));
    assert(x.at({0, 1}) > x.at({0, 0}));

    // 第 1 行全 0 输入 Softmax 后应该是均匀分布 1/3
    assert(approx_equal(x.at({1, 0}), 1.0f / 3.0f));
    assert(approx_equal(x.at({1, 1}), 1.0f / 3.0f));
    assert(approx_equal(x.at({1, 2}), 1.0f / 3.0f));

    std::cout << "PASSED! ✅\n";
}

void test_matmul_op(){
    std::cout << "[Test] Running Matmul Op Test... " << std::flush;

    Tensor A({2, 3});
    A.at({0, 0}) = 1.0f; A.at({0, 1}) = 2.0f; A.at({0, 2}) = 3.0f;
    A.at({1, 0}) = 4.0f; A.at({1, 1}) = 5.0f; A.at({1, 2}) = 6.0f;

    Tensor B({3, 2});
    B.at({0, 0}) = 7.0f;  B.at({0, 1}) = 8.0f;
    B.at({1, 0}) = 9.0f;  B.at({1, 1}) = 10.0f;
    B.at({2, 0}) = 11.0f; B.at({2, 1}) = 12.0f;

    Tensor C({2, 2});
    matmul(A, B, C);

    assert(approx_equal(C.at({0, 0}), 58.0f));
    assert(approx_equal(C.at({0, 1}), 64.0f));
    assert(approx_equal(C.at({1, 0}), 139.0f));
    assert(approx_equal(C.at({1, 1}), 154.0f));

    std::cout << "PASSED! ✅\n";
}

int main(){
    std::cout << "========================================\n";
    std::cout << "   Running Transformer C++ Unit Tests   \n";
    std::cout << "========================================\n";

    test_softmax_op();
    test_matmul_op();

    std::cout << "\nAll active unit tests passed successfully! 🚀\n";
    return 0;
}