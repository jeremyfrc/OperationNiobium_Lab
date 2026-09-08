#include <iostream>
#include <cassert>
#include <vector>
#include <algorithm>
#include "tensor.h"
#include "attention.h"
#include "config.h"
#include "test_utils.h"
#include "ops.h"

// 辅助函数：根据 Python dump 的顺序加载 weights
// 注意：这取决于你在 Python 里如何保存这些数据的顺序
void load_attention_weights(AttentionWeights& w, const std::string& dir) {
    auto wQ_data = loadBinFile(dir + "/attn_wQ.bin");
    auto wK_data = loadBinFile(dir + "/attn_wK.bin");
    auto wV_data = loadBinFile(dir + "/attn_wV.bin");
    auto wO_data = loadBinFile(dir + "/attn_wO.bin");

    // 这里需要根据 AttentionWeights struct 的定义填充 Tensor
    // 假设 Tensor 构造函数支持用 std::vector<float> 初始化
    // 你可能需要修改 Tensor 构造函数来支持这种初始化，或者手动 copy
    // 下面假设 Tensor 有一个 setData 方法或类似机制
    std::copy(wQ_data.begin(), wQ_data.end(), w.wQ.data());
    std::copy(wK_data.begin(), wK_data.end(), w.wK.data());
    std::copy(wV_data.begin(), wV_data.end(), w.wV.data());
    std::copy(wO_data.begin(), wO_data.end(), w.wO.data());
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "       Running Attention Block Test     " << std::endl;
    std::cout << "========================================" << std::endl;

    try {
        // 1. 配置 (应与 dump_reference.py 中的一致)
        TransformerConfig cfg;
        cfg.d_model = 64;
        cfg.n_heads = 4;
        cfg.n_kv_heads = 2;
        cfg.d_head = 16;
        cfg.max_seq_len = 8;
        cfg.rope_theta = 10000.0f;
        cfg.rms_eps = 1e-5f;

        // 2. 加载数据
        std::string data_dir = "tests/ref/data";
        auto x_data = loadBinFile(data_dir + "/attn_input.bin");
        auto ref_out = loadBinFile(data_dir + "/attn_out.bin");

        Tensor x({8, cfg.d_model}, x_data);
        AttentionWeights w {
            Tensor({cfg.d_model, cfg.n_heads * cfg.d_head}),
            Tensor({cfg.d_model, cfg.n_kv_heads * cfg.d_head}),
            Tensor({cfg.d_model, cfg.n_kv_heads * cfg.d_head}),
            Tensor({cfg.n_heads * cfg.d_head, cfg.d_model})
        };
        load_attention_weights(w, data_dir);

        // 3. 执行
        Tensor out({8, cfg.d_model});
        attention_forward(x, w, cfg, out);

        // 4. 校验
        bool ok = check_close(out.data(), ref_out.data(), ref_out.size());
        if (ok) {
            std::cout << "✅ [PASS] Attention block test passed!" << std::endl;
        } else {
            std::cout << "❌ [FAIL] Attention block test failed!" << std::endl;
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
