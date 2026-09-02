#include <iostream>
#include <cmath>
#include <cassert>
#include "tensor.h"
#include "swiglu.h"
#include "test_utils.h" // 引用测试工具头文件

static bool approx_equal(float a, float b, float epsilon = 1e-3f) {
    return std::fabs(a - b) < epsilon;
}

void test_swiglu_op_local() {
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

int test_swiglu_ref() {
    try {
        std::string base = "tests/ref/data/";

        // 1. 文件名修正：与 dump_reference.py 中的 save_bin("swiglu_x.bin", ...) 保持一致
        auto x_data = loadBinFile(base + "swiglu_x.bin");
        auto w_gate_data = loadBinFile(base + "swiglu_w_gate.bin");
        auto w_up_data = loadBinFile(base + "swiglu_w_up.bin");
        auto w_down_data = loadBinFile(base + "swiglu_w_down.bin");
        auto ref_out = loadBinFile(base + "swiglu_out.bin");

        // 2. 根据 tiny_cfg 匹配 Shape
        // (如果 Python 端使用的是 hidden_dim=64, ffn_dim=128, batch=2, seq_len=4):
        int batch = 2, seq_len = 4;
        int in_dim = 64, ffn_dim = 128;
        int num_tokens = batch * seq_len; // 8

        // 如果你的 matmul 只支持 2D 输入，将 input 构造为 [num_tokens, in_dim]
        Tensor input({num_tokens, in_dim}, x_data);
        Tensor w_gate({in_dim, ffn_dim}, w_gate_data);
        Tensor w_up({in_dim, ffn_dim}, w_up_data);
        Tensor w_down({ffn_dim, in_dim}, w_down_data);
        
        Tensor output({num_tokens, in_dim});
        FFNWeights weights(w_gate, w_up, w_down);

        // 3. 执行前向
        swiglu_forward(input, weights, output);

        // 4. 误差断言校验
        bool ok = check_close(output.data(), ref_out.data(), ref_out.size(), 1e-4f, 1e-5f);
        if (ok) {
            std::cout << "✅ [PASS] SwiGLU test passed!" << std::endl;
            return 0;
        } else {
            std::cout << "❌ [FAIL] SwiGLU test failed!" << std::endl;
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}


int main() {
    std::cout << "========================================\n";
    std::cout << "       Running SwiGLU Unit Test         \n";
    std::cout << "========================================\n";

    test_swiglu_op_local();
    std::cout << "\nAll local test cases passed successfully! 🚀\n";

    int s = test_swiglu_ref();
    if (s != 0) {
        std::cerr << "SwiGLU reference test failed!" << std::endl;
        return s; // 返回错误码
    }
    std::cout << "\nAll python ref SwiGLU test cases passed successfully! 🚀\n";
    return 0;
}