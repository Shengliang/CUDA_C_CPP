// gpu_register_test.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>
#include <vector>

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


// Template kernel with UNROLL factor controlling independent FMAs
template<int UNROLL>
__global__ void fma_kernel(float* out, int iters) {
    float x = 1.0f;
    float a = 1.0001f;
    float b = 0.9999f;

    #pragma unroll
    for (int i = 0; i < iters; i++) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            x = fmaf(x, a, b);
        }
    }

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    out[idx] = x;
}

// Run a single FMA test with given thread/block configuration
float run_fma_test(int blocks, int threads, int iters, int unroll_factor) {
    int N = blocks * threads;
    std::vector<float> h_out(N, 0.0f);
    float* d_out = nullptr;
    cudaMalloc(&d_out, N * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Dispatch kernel based on unroll factor
    switch(unroll_factor) {
        case 8: fma_kernel<8><<<blocks, threads>>>(d_out, iters); break;
        case 16: fma_kernel<16><<<blocks, threads>>>(d_out, iters); break;
        case 32: fma_kernel<32><<<blocks, threads>>>(d_out, iters); break;
        case 48: fma_kernel<48><<<blocks, threads>>>(d_out, iters); break;
        case 64: fma_kernel<64><<<blocks, threads>>>(d_out, iters); break;
        case 128: fma_kernel<128><<<blocks, threads>>>(d_out, iters); break;
        case 256: fma_kernel<256><<<blocks, threads>>>(d_out, iters); break;
        default: std::cerr << "Unsupported unroll factor\n"; cudaFree(d_out); return 0; 
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_out);

    // FLOPs = threads * blocks * iters * unroll_factor * 2 (FMA = 2 FLOPs)
    double flops = static_cast<double>(threads) * blocks * iters * unroll_factor * 2.0;
    double tflops = flops / (ms * 1e6); // ms -> s

    std::cout << "Regs: " << unroll_factor
              << ", Threads: " << threads
              << ", Blocks: " << blocks
              << ", Time(ms): " << ms
              << ", TFLOPS: " << tflops << "\n";

    return tflops;
}

// Run all FMA tests
void run_gpu_register_fma_tests() {
    int threads = 512;
    int blocks = 48;
    int iters = 100000; // adjust for ~10-100ms runtime
    std::vector<int> unrolls = {8,16,32,48,64,128,256};

    std::cout << "\n--- Compute-bound GPU FMA (FP32) ---\n";
    for (int u : unrolls) {
        run_fma_test(blocks, threads, iters, u);
    }
}

