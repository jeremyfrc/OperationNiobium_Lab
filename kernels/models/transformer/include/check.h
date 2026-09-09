#pragma once
#include <stdexcept>
#include <string>

//契约级断言： 与assert不同，NB_CHECK不受 -DNDEBUG影响，release build里
//依然生效。 用于shape/layout/numel这类错了必须当场炸的不变量；抛异常
// 可被上层try/catch捕获并打印
// 热路径里的高频、非契约检查仍可用assert
#define NB_CHECK(expr, msg) \
    do { \
        if (!(expr)) { \
            throw std::runtime_error(std::string("NB_CHECK failed: ") + msg); \
        } \
    } while (0)