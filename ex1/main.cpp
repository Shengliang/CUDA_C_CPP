
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

extern "C" void launch_add(
    const int* a, const int* b, int* c, int n,
    cudaEvent_t start, cudaEvent_t stop);

// Problem size
constexpr int N = 1 << 24;       // ~16M elements
constexpr int iterations = 100;   // number of kernel runs for timing

int main()
{
    // Allocate unified memory
    int *a, *b, *c;
    size_t bytes = N * sizeof(int);

    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);

    // Initialize arrays
    for (int i = 0; i < N; ++i) {
        a[i] = i;
        b[i] = 2 * i;
    }

    // Create CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ----------------------
    // Warm-up run
    // ----------------------
    launch_add(a, b, c, N, start, stop);
    cudaDeviceSynchronize();

    // ----------------------
    // Timed runs
    // ----------------------
    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i) {
        launch_add(a, b, c, N, start, stop);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Compute elapsed time per iteration
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= iterations;

    double seconds = ms / 1000.0;

    // Throughput calculations
    double ops = static_cast<double>(N);        // 1 integer add per element
    double gops = ops / seconds / 1e9;         // GigaOps/s
    double bandwidth_gb = 3.0 * bytes / seconds / 1e9; // 3 memory accesses per element (a+b+c)

    // Print results
    std::cout << "Kernel time (per iteration): " << ms << " ms\n";
    std::cout << "Throughput: " << gops << " Gops/s\n";
    std::cout << "Memory Bandwidth: " << bandwidth_gb << " GB/s\n";

    // Synchronize before accessing results
    cudaDeviceSynchronize();
    std::cout << "Sample result c[0] = " << c[0] << ", c[N-1] = " << c[N-1] << "\n";

    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    return 0;
}

