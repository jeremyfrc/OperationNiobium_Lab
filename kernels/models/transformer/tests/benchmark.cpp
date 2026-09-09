#include "transformer.h"
#include "test_utils.h"
#include <iostream>
#include <chrono>
#include <vector>

int main() {
    // 假设你已经有了一个加载权重的逻辑
    TransformerConfig cfg = get_tiny_config();
    TransformerWeights weights = load_tiny_weights(cfg);
    std::vector<int> prompt = {1, 2, 3, 4};
    int max_tokens = 50;

    std::cout << "Starting benchmark for " << max_tokens << " tokens..." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    
    // 运行生成
    auto generated = generate(prompt, weights, cfg, max_tokens);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    double ms_per_token = duration.count() / max_tokens;
    std::cout << "Time per token: " << ms_per_token << " ms" << std::endl;
    std::cout << "Tokens per second: " << 1000.0 / ms_per_token << std::endl;

    return 0;
}