#include "transformer.h"
#include "test_utils.h"
#include "config.h"
#include <iostream>
#include <vector>

int main() {
    // 1. 初始化配置和权重
    TransformerConfig cfg = get_tiny_config();
    TransformerWeights weights = load_tiny_weights(cfg);

    // 2. 输入 Prompt
    std::vector<int> prompt = {1, 5, 42};
    int new_tokens = 5;

    std::cout << "Prompt: ";
    for (int id : prompt) std::cout << id << " ";
    std::cout << std::endl;

    // 3. 执行生成 (普通版 generate：内部逐 token 重算全序列，不使用 KV-Cache)
    auto result = generate(prompt, weights, cfg, new_tokens);

    // 4. 打印结果
    std::cout << "Generated: ";
    for (int id : result) std::cout << id << " ";
    std::cout << std::endl;

    // 5. 简单验证：返回序列长度 = prompt 长度 + 新生成 token 数
    if (result.size() == prompt.size() + (size_t)new_tokens) {
        std::cout << "✅ Generation Test Success!" << std::endl;
    } else {
        std::cout << "❌ Generation Test Failed!" << std::endl;
        return 1;
    }

    return 0;
}