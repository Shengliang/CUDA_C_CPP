cmake_minimum_required(VERSION 3.26)

project(cuda_cpp
    LANGUAGES CXX CUDA
)

# ----------------------------------
# Host C++ (MSVC): C++23
# ----------------------------------
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ----------------------------------
# CUDA: max supported is C++20
# ----------------------------------
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(cuda_cpp
    main.cpp
    kernels.cu
)

# ----------------------------------
# CUDA properties
# ----------------------------------
set_target_properties(cuda_cpp PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES native
)

# ----------------------------------
# Warnings only (NO /std flags here)
# ----------------------------------
if (MSVC)
    target_compile_options(cuda_cpp PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>
    )
endif()

