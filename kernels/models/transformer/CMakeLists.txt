cmake_minimum_required(VERSION 3.16)
project(niobium_transformer_forward CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-Wall -Wextra -O2)

add_library(transformer_forward
    src/tensor.cpp
    src/ops.cpp
    src/rmsnorm.cpp
    src/rope.cpp
    src/attention.cpp
    src/swiglu.cpp
    src/transformer.cpp
)

# 2. 指定头文件目录：PUBLIC 意味着链接这个库的目标（如测试）也能自动使用 include/ 里的头文件
target_include_directories(transformer_forward PUBLIC
    include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common)   # kernels/common (共享 check.h)

# 注: paged_kv_blocks 由顶层 kernels/CMakeLists.txt 的 add_subdirectory(paged_kv) 提供

# ===== 集成层: paged attention 入口 + 对拍测试 (只有这里依赖 paged_kv) =====
add_library(transformer_paged
    src/attention_paged.cpp
)

target_link_libraries(transformer_paged
    PUBLIC transformer_forward
    PUBLIC paged_kv_blocks
)
target_include_directories(transformer_paged PUBLIC include)   # 已由 transformer_forward 带; 保险

# ==============================================================================
# 3. 测试 target 注册（等写到对应 Stage 再取消注释，先放在这里做脚手架）
# ==============================================================================

# Stage 1 单测
add_executable(test_rmsnorm tests/test_rmsnorm.cpp)
target_link_libraries(test_rmsnorm PRIVATE transformer_forward)

add_executable(test_rope tests/test_rope.cpp)
target_link_libraries(test_rope PRIVATE transformer_forward)

add_executable(test_softmax tests/test_softmax.cpp)
target_link_libraries(test_softmax PRIVATE transformer_forward)

add_executable(test_swiglu tests/test_swiglu.cpp)
target_link_libraries(test_swiglu PRIVATE transformer_forward)

# Stage 2 单测
add_executable(test_attention_block tests/test_attention_block.cpp)
target_link_libraries(test_attention_block PRIVATE transformer_forward)

# Stage 3 单测
add_executable(test_full_forward tests/test_full_forward.cpp)
target_link_libraries(test_full_forward PRIVATE transformer_forward)

# Stage 4 单测
add_executable(test_generate tests/test_generate.cpp)
target_link_libraries(test_generate PRIVATE transformer_forward)

# Stage5 KV-Cache 版本的生成测试
add_executable(test_generate_kv tests/test_generate_kv.cpp)
target_link_libraries(test_generate_kv PRIVATE transformer_forward)

# Stage 5 性能benchmark
add_executable(benchmark tests/benchmark.cpp)
target_link_libraries(benchmark PRIVATE transformer_forward)

enable_testing()

add_test(NAME test_rmsnorm        COMMAND test_rmsnorm)
add_test(NAME test_rope           COMMAND test_rope)
add_test(NAME test_softmax        COMMAND test_softmax)
add_test(NAME test_swiglu         COMMAND test_swiglu)
add_test(NAME test_attention_block COMMAND test_attention_block)
add_test(NAME test_full_forward   COMMAND test_full_forward)
add_test(NAME test_generate       COMMAND test_generate)
add_test(NAME test_generate_kv    COMMAND test_generate_kv)
add_test(NAME benchmark           COMMAND benchmark)

# 这些测试用相对路径 "tests/ref/data/*.bin" 读参考数据, 工作目录必须设为
# 源码根(transformer/), 否则 ctest 默认在 build/ 下找不到数据文件。
set(TSFM_TESTS
    test_rmsnorm
    test_rope
    test_softmax
    test_swiglu
    test_attention_block
    test_full_forward
    test_generate
    test_generate_kv
    benchmark)

set_tests_properties(${TSFM_TESTS} PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

