#pragma once
#include <string>
#include <stdexcept>

namespace attn {

struct SchedulerError: std::runtime_error {
    using std::runtime_error::runtime_error;
};

// 块用尽的占位错误: stageD做swap-to-CPU/ 抢占重算；此前一律可控报错
class OutOfBlocksError: public std::runtime_error {
    public:
        explicit OutOfBlocksError(const std::string& what) : std::runtime_error(what) {}
};
}
