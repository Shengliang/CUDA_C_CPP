
#include <cuda_runtime.h>
#include <cstdio>

constexpr int threads = 256;
constexpr int blocks = 256;

// -------------------------------
// Memory-bound ADD kernel
// -------------------------------
__global__ void memory_add_kernel(const int* a, const int* b, int* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

void launch_memory_add(const int* d_a, const int* d_b, int* d_c, int n) {
    int blocks_mem = (n + threads - 1) / threads;
    memory_add_kernel<<<blocks_mem, threads>>>(d_a, d_b, d_c, n);
}

// -------------------------------
// Compute-bound ADD kernel
// -------------------------------
__global__ void compute_add_kernel(int repeats) {
    int x = threadIdx.x;
    int y = threadIdx.y;

    #pragma unroll
    for (int i = 0; i < repeats; ++i) {
        x += 1;
        y += 2;
        x += y;
        y += x;
    }

    // Prevent compiler optimization
    if (x + y == 0) printf("");
}

void launch_compute_add(int repeats) {
    compute_add_kernel<<<blocks, threads>>>(repeats);
}

