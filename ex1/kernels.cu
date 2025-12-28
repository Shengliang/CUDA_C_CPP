

#include <cuda_runtime.h>

__global__ void add_kernel(const int* a, const int* b, int* c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        c[idx] = a[idx] + b[idx];
}

extern "C" void launch_add(
    const int* a, const int* b, int* c, int n)
{
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    add_kernel<<<blocks, threads>>>(a, b, c, n);
}
