#include "common.h"
#include "gpu_register_test.h"
#include <cuda_runtime.h>
#include <cstdio>

template<int REG>
__global__ void compute_add_kernel(int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    int r[REG];
    #pragma unroll
    for (int i = 0; i < REG; i++)
        r[i] = idx + i;

    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < N; i += stride) {
        #pragma unroll
        for (int j = 0; j < REG; j++)
            r[j] += 1;
    }

    int tmp = 0;
    #pragma unroll
    for (int j = 0; j < REG; j++)
        tmp += r[j];

    volatile int sink = tmp;   // single volatile store
}

void run_compute_benchmark(int N){
    int threads=512, blocks=48;
    int regs[]={8,16,32,48,64,128,256};

    printf("\n--- Compute-bound GPU ADD ---\n");
    printf("    Regs  Threads   Blocks   Time(ms)       Gops/s\n");

    for(int r=0;r<sizeof(regs)/sizeof(int);r++){
        float ms;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        switch(regs[r]){
            case 8:  compute_add_kernel<8><<<blocks,threads>>>(N); break;
            case 16: compute_add_kernel<16><<<blocks,threads>>>(N); break;
            case 32: compute_add_kernel<32><<<blocks,threads>>>(N); break;
            case 48: compute_add_kernel<48><<<blocks,threads>>>(N); break;
            case 64: compute_add_kernel<64><<<blocks,threads>>>(N); break;
            case 128: compute_add_kernel<128><<<blocks,threads>>>(N); break;
            case 256: compute_add_kernel<256><<<blocks,threads>>>(N); break;
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms,start,stop);

        double gops=double(N)/(ms/1000.0)/1e9;
        printf("%8d %8d %8d %10.3f %12.2f\n",regs[r],threads,blocks,ms,gops);
    }
}

