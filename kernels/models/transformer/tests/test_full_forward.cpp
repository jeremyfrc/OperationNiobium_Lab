#include "transformer.h"
#include "test_utils.h"
#include <iostream>
#include <vector>


// 一个辅助函数，简化 Tensor 的创建和数据填充
Tensor load_to_tensor(const std::string& path, std::vector<int> shape) {
    auto data = loadBinFile(path);
    Tensor t(shape);
    // 检查数据大小是否匹配
    if (data.size() != t.numel()) {
        throw std::runtime_error("Size mismatch for " + path);
    }
    std::copy(data.begin(), data.end(), t.data());
    return t;
}

int main() {
    // 1. 定义与 Python 侧一致的配置
    TransformerConfig cfg;
    cfg.vocab_size = 50;
    cfg.d_model = 16;
    cfg.n_layers = 2;
    cfg.n_heads = 4;
    cfg.n_kv_heads = 2;
    cfg.d_head = 4;
    cfg.d_ff = 32;
    cfg.max_seq_len = 8;
    cfg.rms_eps = 1e-5f;
    cfg.rope_theta = 10000.0f;

    TransformerWeights weights;
    std::string base = "tests/ref/data/full_";

    // 2. 加载全局权重
    weights.token_embedding = load_to_tensor(base + "embed.bin", {cfg.vocab_size, cfg.d_model});
    weights.final_rms_weight = load_to_tensor(base + "final_rms.bin", {cfg.d_model});
    weights.lm_head = load_to_tensor(base + "lm_head.bin", {cfg.d_model, cfg.vocab_size});

    for (int i = 0; i < cfg.n_layers; ++i) {
        DecoderLayerWeights l;
        std::string p = base + "layer" + std::to_string(i) + "_";
        
        l.rms1_weight = load_to_tensor(p + "rms1.bin", {cfg.d_model});
        l.attn.wQ = load_to_tensor(p + "wQ.bin", {cfg.d_model, cfg.n_heads * cfg.d_head});
        l.attn.wK = load_to_tensor(p + "wK.bin", {cfg.d_model, cfg.n_kv_heads * cfg.d_head});
        l.attn.wV = load_to_tensor(p + "wV.bin", {cfg.d_model, cfg.n_kv_heads * cfg.d_head});
        l.attn.wO = load_to_tensor(p + "wO.bin", {cfg.n_heads * cfg.d_head, cfg.d_model});
        l.rms2_weight = load_to_tensor(p + "rms2.bin", {cfg.d_model});
        l.ffn.wGate = load_to_tensor(p + "w_gate.bin", {cfg.d_model, cfg.d_ff});
        l.ffn.wUp   = load_to_tensor(p + "w_up.bin", {cfg.d_model, cfg.d_ff});
        l.ffn.wDown = load_to_tensor(p + "w_down.bin", {cfg.d_ff, cfg.d_model});
        
        weights.layers.push_back(l);
    }

    // 3. 执行推理
    std::vector<int> input_ids = {1, 5, 42, 7};
    Tensor logits({(int)input_ids.size(), cfg.vocab_size});
    transformer_forward(input_ids, weights, cfg, logits);

    // 4. 对比参考值
    auto ref_logits = loadBinFile("tests/ref/data/full_logits_ref.bin");
    if (check_close(logits.data(), ref_logits.data(), logits.numel())) {
        std::cout << "✅ Stage 3: Full Forward Test Passed!" << std::endl;
    } else {
        std::cout << "❌ Stage 3: Full Forward Test Failed!" << std::endl;
    }

    return 0;
}