cmake_minimum_required(VERSION 3.16)
project(niobium_attention_blocks CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-Wall -Wextra -O2)

add_library(attn_blocks
    src/block_allocator.cpp
    src/block_table.cpp
    src/paged_kv_cache.cpp
)
target_include_directories(attn_blocks PUBLIC
    include
    ${CMAKE_CURRENT_SOURCE_DIR}/../common)   # kernels/common (共享 check.h)

# ===== 单测 target =====
add_executable(test_block_allocator tests/test_block_allocator.cpp)
target_link_libraries(test_block_allocator PRIVATE attn_blocks)

add_executable(test_allocate_table tests/test_allocate_table.cpp)
target_link_libraries(test_allocate_table PRIVATE attn_blocks)

add_executable(test_paged_kv_cache tests/test_paged_kv_cache.cpp)
target_link_libraries(test_paged_kv_cache PRIVATE attn_blocks)

enable_testing()
add_test(NAME test_block_allocator COMMAND test_block_allocator)
add_test(NAME test_allocate_table COMMAND test_allocate_table)
add_test(NAME test_paged_kv_cache COMMAND test_paged_kv_cache)