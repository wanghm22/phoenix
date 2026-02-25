cmake_minimum_required(VERSION 3.18)
project(libphoenix-benchmarks)

# 查找 CUDA
find_package(CUDA 12.4 REQUIRED)

# 使用 make_benchmark 宏创建可执行文件
make_benchmark(
    hnsw_search                  # target 名称
    hnsw_search                  # binary_name 输出文件名
    hnsw_vector_search.cu        # 源文件列表
)

# 设置 CUDA 相关属性
set_target_properties(hnsw_search PROPERTIES
    CUDA_ARCHITECTURES "80"                  # 根据您的 GPU 修改
    CUDA_RUNTIME_LIBRARY "Shared"
    CUDA_RESOLVE_DEVICE_SYMBOLS ON           # 启用设备符号解析
    CUDA_SEPARABLE_COMPILATION ON            # 启用可分编译
)

# 直接设置 CMAKE_CUDA_FLAGS（全局方式）
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true -O3")