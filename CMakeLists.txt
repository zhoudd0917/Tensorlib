cmake_minimum_required(VERSION 3.18)
project(TensorLib LANGUAGES CXX CUDA)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

enable_language(CUDA)

set(TARGET_NAME tensorlib)

set(CUDA_SOURCES src/my_cuda_kernel.cu)

pybind11_add_module(${TARGET_NAME} src/pybind_wrapper.cpp ${CUDA_SOURCES})

set_target_properties(${TARGET_NAME} PROPERTIES CUDA_ARCHITECTURES 75)

set_target_properties(${TARGET_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(${TARGET_NAME} PRIVATE CUDA::cudart CUDA::cublas)

target_include_directories(${TARGET_NAME} PRIVATE ${Python3_INCLUDE_DIRS} ${pybind11_INCLUDE_DIRS})