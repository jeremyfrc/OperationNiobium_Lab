#include <iostream>
#include <cmath>
#include <cassert>
#include "tensor.h"
#include "swiglu.h"

static bool approx_equal(float a, float b, float epsilon = 1e-3f) {
    return std::fabs(a - b) < epsilon;
}

void test_swiglu_op() {
    std::cout << "[Test] Running SwiGLU Forward FFN Test... " << std::flush;

    // 参数设定: seq_len=1, hidden_dim=2, intermediate_dim=2
    Tensor x({1, 2});
    x.at({0, 0}) = 1.0f; 
    x.at({0, 1}) = 2.0f;

    Tensor w_gate({2,2});
    Tensor w_up({2, 2});
    Tensor w_down({2, 2});

    for (int i = 0; i < 4; ++i) {
        w_gate.data()[i] = 1.0f;
        w_up.data()[i]   = 1.0f;
        w_down.data()[i] = 1.0f;
    }
    FFNWeights w(w_gate, w_up, w_down);

    // 简化权重赋值（单位矩阵或常数）
    // W_gate = [[1, 0], [0, 1]] -> gate_proj = [1.0, 2.0]
    w.wGate.at({0, 0}) = 1.0f; w.wGate.at({0, 1}) = 0.0f;
    w.wGate.at({1, 0}) = 0.0f; w.wGate.at({1, 1}) = 1.0f;

    // W_up = [[0.5, 0], [0, 0.5]] -> up_proj = [0.5, 1.0]
    w.wUp.at({0, 0}) = 0.5f; w.wUp.at({0, 1}) = 0.0f;
    w.wUp.at({1, 0}) = 0.0f; w.wUp.at({1, 1}) = 0.5f;

    // W_down = [[1, 0], [0, 1]]
    w.wDown.at({0, 0}) = 1.0f; w.wDown.at({0, 1}) = 0.0f;
    w.wDown.at({1, 0}) = 0.0f; w.wDown.at({1, 1}) = 1.0f;

    Tensor out({1, 2});
    swiglu_forward(x, w, out);

    // 手动验证计算逻辑:
    // gate_proj = [1.0, 2.0]
    // up_proj   = [0.5, 1.0]
    // silu(1.0) = 1.0 / (1 + exp(-1)) ≈ 0.7310586
    // silu(2.0) = 2.0 / (1 + exp(-2)) ≈ 1.7615941
    // intermediate[0] = 0.7310586 * 0.5 = 0.3655293
    // intermediate[1] = 1.7615941 * 1.0 = 1.7615941
    // out = intermediate @ I = [0.3655293, 1.7615941]

    float exp_inter0 = (1.0f / (1.0f + std::exp(-1.0f))) * 0.5f;
    float exp_inter1 = (2.0f / (1.0f + std::exp(-2.0f))) * 1.0f;

    assert(approx_equal(out.at({0, 0}), exp_inter0));
    assert(approx_equal(out.at({0, 1}), exp_inter1));

    std::cout << "PASSED! ✅\n";
}


int main() {
    std::cout << "========================================\n";
    std::cout << "       Running SwiGLU Unit Test         \n";
    std::cout << "========================================\n";

    test_swiglu_op();

    std::cout << "\nAll test cases passed successfully! 🚀\n";
    return 0;
}