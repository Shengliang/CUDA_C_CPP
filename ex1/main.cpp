#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

extern "C" void launch_add(const int* d_a, const int* d_b, int* d_c, int n);

// Problem size
constexpr int N = 1 << 24;       // 16M elements
constexpr int iterations = 100;   // number of kernel runs for timing

// Helper macro to check CUDA errors
#define CUDA_CHECK(call)                                                      \
do {                                                                          \
    cudaError_t err = call;                                                   \
    if(err != cudaSuccess) {                                                  \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err)                \
                  << " at line " << __LINE__ << std::endl;                   \
        exit(EXIT_FAILURE);                                                   \
    }                                                                         \
} while(0)

int main() {
    // -------------------------------
    // Allocate host memory (pinned for high bandwidth)
    // -------------------------------
    int *h_a, *h_b, *h_c;
    size_t bytes = N * sizeof(int);
    CUDA_CHECK(cudaMallocHost(&h_a, bytes));  // pinned
    CUDA_CHECK(cudaMallocHost(&h_b, bytes));
    CUDA_CHECK(cudaMallocHost(&h_c, bytes));

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_a[i] = i;
        h_b[i] = 2 * i;
    }

    // -------------------------------
    // Allocate device memory
    // -------------------------------
    int *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    // -------------------------------
    // Create CUDA events
    // -------------------------------
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // -------------------------------
    // Copy data to device
    // -------------------------------
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // -------------------------------
    // Warm-up kernel
    // -------------------------------
    launch_add(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // -------------------------------
    // Timed kernel launches
    // -------------------------------
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        launch_add(d_a, d_b, d_c, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    ms /= iterations;  // average per iteration

    double seconds = ms / 1000.0;

    // -------------------------------
    // Copy result back to host
    // -------------------------------
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    // -------------------------------
    // Throughput calculations
    // -------------------------------
    double ops = static_cast<double>(N);        // 1 addition per element
    double gops = ops / seconds / 1e9;

    // Memory bandwidth: read a + read b + write c
    double bandwidth_gb = 3.0 * bytes / seconds / 1e9;

    // Print results
    std::cout << "Kernel time (per iteration): " << ms << " ms\n";
    std::cout << "Throughput: " << gops << " Gops/s\n";
    std::cout << "Memory Bandwidth: " << bandwidth_gb << " GB/s\n";
    std::cout << "Sample result c[0] = " << h_c[0] << ", c[N-1] = " << h_c[N-1] << "\n";

    // -------------------------------
    // Optional: Theoretical peak placeholders
    // -------------------------------
    const double peak_gops = 2.3e5;      // example placeholder, RTX 5070 ~ 230 TFLOPS int32
    const double peak_bw = 900.0;        // example placeholder, GB/s

    std::cout << "Theoretical peak compute: " << peak_gops << " Gops/s\n";
    std::cout << "Theoretical peak memory:  " << peak_bw << " GB/s\n";

    // -------------------------------
    // Cleanup
    // -------------------------------
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFreeHost(h_a));
    CUDA_CHECK(cudaFreeHost(h_b));
    CUDA_CHECK(cudaFreeHost(h_c));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}

