
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include "kernels.cuh"   // CUDA kernels header

// -------------------------------
// Configurable parameters
// -------------------------------
constexpr int N = 1 << 24;           // 16M elements
constexpr int iterations = 100;      // memory-bound kernel iterations
constexpr int compute_repeats = 1000; // repeated ADDs per thread

// -------------------------------
// CUDA error checking
// -------------------------------
#define CUDA_CHECK(call)                                                      \
do {                                                                          \
    cudaError_t err = call;                                                   \
    if(err != cudaSuccess) {                                                  \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err)                \
                  << " at line " << __LINE__ << std::endl;                   \
        exit(EXIT_FAILURE);                                                   \
    }                                                                         \
} while(0)

// -------------------------------
// Benchmark helper
// -------------------------------
template<typename Func>
float benchmark_kernel(Func kernel, int repeat = 1) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeat; ++i) kernel();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / repeat;  // average per iteration
}


// -------------------------------
// Main program
// -------------------------------
int main() {
    std::cout << "Running 2 tests: Memory-bound and Compute-bound ADD kernels\n";

    // --- Memory-bound kernel ---
    std::cout << "\n--- Memory-bound ADD Kernel ---\n";
    size_t bytes = N * sizeof(int);

    int *h_a, *h_b, *h_c;
    CUDA_CHECK(cudaMallocHost(&h_a, bytes));
    CUDA_CHECK(cudaMallocHost(&h_b, bytes));
    CUDA_CHECK(cudaMallocHost(&h_c, bytes));

    for (int i = 0; i < N; ++i) {
        h_a[i] = i;
        h_b[i] = 2*i;
    }

    int *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    launch_memory_add(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    float ms_mem = benchmark_kernel([&](){ launch_memory_add(d_a, d_b, d_c, N); }, iterations);

    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    double ops_mem = static_cast<double>(N);
    double gops_mem = ops_mem / (ms_mem / 1000.0) / 1e9;
    double bw_mem = 3.0 * bytes / (ms_mem / 1000.0) / 1e9;

    std::cout << "Kernel time (per iteration): " << ms_mem << " ms\n";
    std::cout << "Throughput: " << gops_mem << " Gops/s\n";
    std::cout << "Memory Bandwidth: " << bw_mem << " GB/s\n";
    std::cout << "Sample result c[0] = " << h_c[0] << ", c[N-1] = " << h_c[N-1] << "\n";

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFreeHost(h_a));
    CUDA_CHECK(cudaFreeHost(h_b));
    CUDA_CHECK(cudaFreeHost(h_c));

    // --- Compute-bound kernel ---
    std::cout << "\n--- Compute-bound ADD Kernel ---\n";
    float ms_comp = benchmark_kernel([&](){ launch_compute_add(compute_repeats); }, 10);

    long long total_ops = 1LL * threads * blocks * compute_repeats * 4;
    double gops_comp = total_ops / (ms_comp / 1000.0) / 1e9;

    std::cout << "Kernel time (per iteration): " << ms_comp << " ms\n";
    std::cout << "Compute-bound throughput: " << gops_comp << " Gops/s\n";

    return 0;
}
