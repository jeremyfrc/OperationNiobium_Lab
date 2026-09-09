#include "transformer.h"
#include "test_utils.h"
#include <iostream>
#include <vector>

int main() {
    // 1. 初始化
    TransformerConfig cfg = get_tiny_config();
    TransformerWeights weights = load_tiny_weights(cfg);

    // 2. 输入 Prompt
    std::vector<int> prompt = {1, 5, 42};
    int new_tokens = 5;

    std::cout << "Prompt: ";
    for(int id : prompt) std::cout << id << " ";
    std::cout << std::endl;

    // 3. 执行生成
    auto result = generate(prompt, weights, cfg, new_tokens);

    // 4. 打印结果
    std::cout << "Generated: ";
    for(int id : result) std::cout << id << " ";
    std::cout << std::endl;

    // 5. 简单验证
    if (result.size() == prompt.size() + new_tokens) {
        std::cout << "✅ Generation Test Success!" << std::endl;
    }

    return 0;
}