#pragma once
#include <stdexcept>
#include <string>

// 契约级断言: 失败抛 std::runtime_error (不受 -DNDEBUG 影响)
#define NB_CHECK(expr, msg)                                              \
    do {                                                                 \
        if (!(expr)) {                                                   \
            throw std::runtime_error(std::string("NB_CHECK failed: ")    \
                                     + (msg) + " | " #expr);             \
        }                                                                \
    } while (0)