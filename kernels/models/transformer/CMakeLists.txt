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
target_include_directories(transformer_forward PUBLIC include)

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
# add_executable(test_attention_block tests/test_attention_block.cpp)
# target_link_libraries(test_attention_block PRIVATE transformer_forward)

# Stage 3 单测
# add_executable(test_full_forward tests/test_full_forward.cpp)
# target_link_libraries(test_full_forward PRIVATE transformer_forward)
