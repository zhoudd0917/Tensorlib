# CMakeLists.txt for TensorLib project
# Compiles for library for both C++ and Python
cmake_minimum_required(VERSION 3.18)
project(TensorLib LANGUAGES CXX CUDA)

enable_language(CUDA)

set(CMAKE_CXX_STANDARD 17)

set(PYTHON_MODULE_NAME tensorlib)
set(CPP_LIBRARY_NAME tensorlib_cpp)

set(CUDA_SOURCES src/tensor.cu src/cublas_handler.cu src/node.cu src/operators.cu)
set(CPP_SOURCES src/autograd.cpp src/cpu_handler.cpp src/utils.cpp src/tensor_factory.cpp)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(OpenBLAS REQUIRED)

pybind11_add_module(${PYTHON_MODULE_NAME} src/pybind_wrapper.cpp ${CUDA_SOURCES} ${CPP_SOURCES})

set_target_properties(${PYTHON_MODULE_NAME} PROPERTIES 
    CUDA_ARCHITECTURES 75
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(${PYTHON_MODULE_NAME} PRIVATE CUDA::cudart CUDA::cublas ${OpenBLAS_LIBRARIES})
target_include_directories(${PYTHON_MODULE_NAME} PRIVATE 
    ${Python3_INCLUDE_DIRS} 
    ${pybind11_INCLUDE_DIRS} 
    ${OpenBLAS_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
)

add_library(${CPP_LIBRARY_NAME} STATIC ${CUDA_SOURCES} ${CPP_SOURCES})

set_target_properties(${CPP_LIBRARY_NAME} PROPERTIES 
    CUDA_ARCHITECTURES 75
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(${CPP_LIBRARY_NAME} PRIVATE CUDA::cudart CUDA::cublas ${OpenBLAS_LIBRARIES})
target_include_directories(${CPP_LIBRARY_NAME} PRIVATE ${OpenBLAS_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)
