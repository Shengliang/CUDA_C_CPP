
// cpu_add_test.cpp
#include <cstdio>
#include <chrono>
#include <vector>
#include <cstdint>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

void run_cpu_add_test()
{
    constexpr size_t N = 1ull << 26;   // 67M elements
    constexpr int ITERS = 50;

    std::vector<float> a(N, 1.0f);
    std::vector<float> b(N, 2.0f);
    std::vector<float> c(N, 0.0f);

    // --- Warm-up ---
    for (size_t i = 0; i < N; i++) {
        c[i] = a[i] + b[i];
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int it = 0; it < ITERS; it++) {
        for (size_t i = 0; i < N; i++) {
            c[i] = a[i] + b[i];
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    double ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count() / ITERS;

    double ops = double(N);
    double gops = ops / (ms * 1e6);

    // Prevent optimization
    volatile float sink = c[0];
    (void)sink;

    printf("\n--- CPU ADD Benchmark ---\n");
    printf("Array size: %zu elements\n", N);
    printf("Kernel time (ms): %7.3f, Throughput: %7.3f Gops/s\n",
           ms, gops);
    printf("Sample result c[0]=%.0f, c[N-1]=%.0f\n", c[0], c[N - 1]);
}

