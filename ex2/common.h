
#pragma once
#include <cstdio>
#include <cuda_runtime.h>

inline void checkCuda(cudaError_t err, const char* msg="") {
    if(err != cudaSuccess){
        printf("CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(-1);
    }
}

