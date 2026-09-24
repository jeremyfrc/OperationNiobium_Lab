cmake_minimum_required(VERSION 3.16)
project(niobium_lab CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra -O2)

enable_testing()   # 必须在 add_subdirectory 之前: 否则顶层无 CTestTestfile, ctest 找不到子目录测试

add_subdirectory(paged_kv)
add_subdirectory(models/transformer)