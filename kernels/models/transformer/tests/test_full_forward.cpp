#include "transformer.h"
#include "test_utils.h"
#include <iostream>
#include <vector>


int main() {
    // 1. 定义与 Python 侧一致的配置
    TransformerConfig cfg = get_tiny_config();

    // 2. 加载权重
    TransformerWeights weights = load_tiny_weights(cfg);

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