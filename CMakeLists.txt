cmake_minimum_required(VERSION 3.12) # 确保使用的是一个较新版本的CMake
project(CudaExamples)

# 设置CUDA编译器和其他必要的设置
find_package(CUDA REQUIRED)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

# CUDA编译标志，根据需要进行调整
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -gencode arch=compute_60,code=sm_60)

# 指定include目录
include_directories(${PROJECT_SOURCE_DIR}/include)

# 定义输出二进制文件的路径
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build_cuda/bin)

# 遍历example目录下的所有子目录，并为每个找到的.cu文件添加一个可执行目标
file(GLOB_RECURSE CU_FILES "example/**/*.cu")
foreach(CU_FILE IN LISTS CU_FILES)
    get_filename_component(EX_NAME ${CU_FILE} NAME_WE) # 获取文件名作为可执行文件的名字
    cuda_add_executable(${EX_NAME} ${CU_FILE})
endforeach()


set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -gencode arch=compute_60,code=sm_60;-allow-unsupported-compiler)

# 如果你有其他源文件或者库需要链接，可以在下面添加
# target_link_libraries(${EX_NAME} otherLibs)

