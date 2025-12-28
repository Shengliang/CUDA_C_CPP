
#include "common.h"
#include "gpu_memory_test.h"
#include <cuda_runtime.h>
#include <cstdio>

__global__ void memory_add_kernel(const int* a, const int* b, int* c, int N, int repeats) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    size_t stride = blockDim.x * gridDim.x;
    for(int r=0; r<repeats; r++){
        for(size_t i=idx;i<N;i+=stride){
            c[i] = a[i]+b[i];
        }
    }
}

void run_memory_benchmark(int N){
    int *a,*b,*c;
    checkCuda(cudaMalloc(&a,N*sizeof(int)),"Alloc a");
    checkCuda(cudaMalloc(&b,N*sizeof(int)),"Alloc b");
    checkCuda(cudaMalloc(&c,N*sizeof(int)),"Alloc c");
    checkCuda(cudaMemset(a,1,N*sizeof(int)),"Memset a");
    checkCuda(cudaMemset(b,2,N*sizeof(int)),"Memset b");

    dim3 threads(512);
    dim3 blocks((N+threads.x-1)/threads.x);
    int repeats=32;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    memory_add_kernel<<<blocks,threads>>>(a,b,c,N,repeats);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms,start,stop);
    double gops = double(N)*repeats/(ms/1000.0)/1e9;

    int first,last;
    checkCuda(cudaMemcpy(&first,c,sizeof(int),cudaMemcpyDeviceToHost),"Copy first");
    checkCuda(cudaMemcpy(&last,c+N-1,sizeof(int),cudaMemcpyDeviceToHost),"Copy last");

    printf("\n--- Memory-bound GPU ADD ---\n");
    printf("Array size: %zu elements\n",N);
    printf("Kernel time (ms): %10.3f, Throughput: %7.3f Gops/s\n",ms,gops);
    printf("Sample result c[0]=%d, c[N-1]=%d\n",first,last);

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
}

