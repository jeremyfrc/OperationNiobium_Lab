#pragma once
#include "transformer.h" 
#include "utils.h"
#include <fstream>
#include <vector>
#include <cmath>
#include <iostream>
#include <string>
#include <stdexcept>

// 1. loader: load .bin file to std::vector<float>
inline std::vector<float> loadBinFile(const std::string& filepath){
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file.is_open()){
        throw std::runtime_error("Failed to open file: " + filepath);
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<float> buffer(size / sizeof(float));
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        throw std::runtime_error("Failed to read binary data from: " + filepath);
    }

    return buffer;
}


// 2. Checker: compare C++ and PYthon ref results
inline bool check_close(const float* actual, const float* expected, size_t size, float rtol = 1e-4f, float atol = 1e-5f) {
    size_t errors = 0;
    for (size_t i = 0; i < size; ++i) {
        float diff = std::abs(actual[i] - expected[i]);
        float max_error = atol + rtol * std::abs(expected[i]);

        if (diff > max_error) {
            if (errors < 5) {
                std::cerr << "Mismatch at index " << i 
                << "; actual= " << actual[i] 
                << ", expected=" << expected[i] 
                << ", diff=" << diff << std::endl;
            }
            errors++;
        }
    }

    if (errors > 0){
        std::cerr << "Total mismatches: " << errors << " /  " << size << std::endl;
        return false;
    }
    return true;
}


inline TransformerConfig get_tiny_config() {
    return {
        .vocab_size = 50, .d_model = 16, .n_layers = 2,
        .n_heads = 4, .n_kv_heads = 2, .d_head = 4,
        .d_ff = 32, .max_seq_len = 8, .rope_theta = 10000.0f, .rms_eps = 1e-5f
    };
}

// 一个辅助函数，简化 Tensor 的创建和数据填充
inline Tensor load_to_tensor(const std::string& path, std::vector<int> shape) {
    auto data = loadBinFile(path);
    Tensor t(shape);
    // 检查数据大小是否匹配
    if (data.size() != t.numel()) {
        throw std::runtime_error("Size mismatch for " + path);
    }
    std::copy(data.begin(), data.end(), t.data());
    return t;
}

inline TransformerWeights load_tiny_weights(
    const TransformerConfig& cfg,
    const std::string& base_dir = "tests/ref/data/full_")
{
    auto loadTensor = [](const std::string& path, std::vector<int> shape) {
        return load_to_tensor(path, shape);
    };

    TransformerWeights w;
    w.token_embedding  = loadTensor(base_dir + "embed.bin", {cfg.vocab_size, cfg.d_model});
    w.final_rms_weight = loadTensor(base_dir + "final_rms.bin", {cfg.d_model});
    w.lm_head          = loadTensor(base_dir + "lm_head.bin", {cfg.d_model, cfg.vocab_size});

    for (int i = 0; i < cfg.n_layers; ++i) {
        std::string p = base_dir + "layer" + std::to_string(i) + "_";
        DecoderLayerWeights l;
        l.rms1_weight      = loadTensor(p + "rms1.bin", {cfg.d_model});
        l.attn.wQ          = loadTensor(p + "wQ.bin",  {cfg.d_model, cfg.n_heads * cfg.d_head});
        l.attn.wK          = loadTensor(p + "wK.bin",  {cfg.d_model, cfg.n_kv_heads * cfg.d_head});
        l.attn.wV          = loadTensor(p + "wV.bin",  {cfg.d_model, cfg.n_kv_heads * cfg.d_head});
        l.attn.wO          = loadTensor(p + "wO.bin",  {cfg.n_heads * cfg.d_head, cfg.d_model});
        l.rms2_weight      = loadTensor(p + "rms2.bin", {cfg.d_model});
        l.ffn.wGate        = loadTensor(p + "w_gate.bin", {cfg.d_model, cfg.d_ff});
        l.ffn.wUp          = loadTensor(p + "w_up.bin",   {cfg.d_model, cfg.d_ff});
        l.ffn.wDown        = loadTensor(p + "w_down.bin", {cfg.d_ff, cfg.d_model});
        w.layers.push_back(l);
    }
    return w;
}
