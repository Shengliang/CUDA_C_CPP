
cmake_minimum_required(VERSION 3.22)
project(cuda_cpp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(cuda_cpp
    main.cpp
    gpu_memory_test.cu
    gpu_register_test.cu
    cpu_add_test.cpp
)

set_target_properties(cuda_cpp PROPERTIES
    CUDA_STANDARD 20
    CUDA_SEPARABLE_COMPILATION ON
)

