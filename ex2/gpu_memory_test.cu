
// gpu_memory_test.cu
#include <cuda_runtime.h>
#include <cstdio>

__global__ void mem_add_kernel(const int* a, const int* b, int* c, size_t N)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < N; i += stride) {
        c[i] = a[i] + b[i];
    }
}

void run_gpu_memory_test()
{
    constexpr size_t N = 1ull << 26;  // 67,108,864 ints (~256 MB total traffic)
    constexpr int THREADS = 256;
    constexpr int BLOCKS  = 256;
    constexpr int ITERS   = 50;

    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N * sizeof(int));
    cudaMalloc(&d_b, N * sizeof(int));
    cudaMalloc(&d_c, N * sizeof(int));

    cudaMemset(d_a, 1, N * sizeof(int));
    cudaMemset(d_b, 2, N * sizeof(int));

    // --- Warm-up (critical) ---
    mem_add_kernel<<<BLOCKS, THREADS>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < ITERS; i++) {
        mem_add_kernel<<<BLOCKS, THREADS>>>(d_a, d_b, d_c, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= ITERS;

    double ops = double(N);
    double gops = ops / (ms * 1e6);

    printf("\n--- Memory-bound GPU ADD ---\n");
    printf("Array size: %zu elements\n", N);
    printf("Kernel time (ms): %7.3f, Throughput: %7.2f Gops/s\n",
           ms, gops);

    int h0, h1;
    cudaMemcpy(&h0, d_c, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h1, d_c + N - 1, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Sample result c[0]=%d, c[N-1]=%d\n", h0, h1);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

