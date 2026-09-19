#pragma once
#include <cstdio>
#include <cstdlib>

namespace attn {

// NB_CHECK(cond, msg): cond 为真则通过；为假打印位置+消息并 abort。
// 常用于两处：
//   (1) 占位——未实现的方法体写 NB_CHECK(false, "todo")；
//   (2) 不变式断言。
#define NB_CHECK(cond, msg)                                              \
  do {                                                                   \
    if (!(cond)) {                                                       \
      std::fprintf(stderr, "[NB_CHECK failed] %s:%d: %s | " #cond "\n",  \
                   __FILE__, __LINE__, (msg));                           \
      std::abort();                                                      \
    }                                                                    \
  } while (0)

} // namespace attn