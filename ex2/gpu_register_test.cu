// gpu_register_test.cu
#include <cuda_runtime.h>
#include <cstdio>

template<int REG>
__global__ void compute_add_kernel()
{
    float r[REG];
    #pragma unroll
    for (int i = 0; i < REG; i++) r[i] = 1.0f;

    constexpr int INNER_ITERS = 4096;

    #pragma unroll 1
    for (int it = 0; it < INNER_ITERS; it++) {
        #pragma unroll
        for (int j = 0; j < REG; j++) {
            r[j] += 1.0f;
        }
    }

    // Prevent dead-code elimination
    volatile float sink = r[0];
    (void)sink;
}

template<int REG>
void run_compute_test(int threads, int blocks)
{
    constexpr int ITERS = 100;

    // Warm-up
    compute_add_kernel<REG><<<blocks, threads>>>();
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < ITERS; i++) {
        compute_add_kernel<REG><<<blocks, threads>>>();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= ITERS;

    double adds =
        double(blocks) * threads *
        REG * 4096;  // INNER_ITERS

    double gops = adds / (ms * 1e6);

    printf("%8d %8d %8d %10.3f %12.2f\n",
           REG, threads, blocks, ms, gops);
}

void run_gpu_register_tests()
{
    const int THREADS = 512;
    const int BLOCKS  = 48;

    printf("\n--- Compute-bound GPU ADD ---\n");
    printf("%8s %8s %8s %10s %12s\n",
           "Regs", "Threads", "Blocks", "Time(ms)", "Gops/s");

    run_compute_test<8>(THREADS, BLOCKS);
    run_compute_test<16>(THREADS, BLOCKS);
    run_compute_test<32>(THREADS, BLOCKS);
    run_compute_test<48>(THREADS, BLOCKS);
    run_compute_test<64>(THREADS, BLOCKS);
    run_compute_test<128>(THREADS, BLOCKS);
    run_compute_test<256>(THREADS, BLOCKS);
}

