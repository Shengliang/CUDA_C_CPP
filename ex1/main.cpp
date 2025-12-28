#include <cuda_runtime.h>
#include <iostream>
#include <span>
#include <vector>
#include <ranges>

extern "C" void launch_add(
    const int*, const int*, int*, int);

constexpr int N = 1 << 20;

int main()
{
    int *a, *b, *c;
    cudaMallocManaged(&a, N * sizeof(int));
    cudaMallocManaged(&b, N * sizeof(int));
    cudaMallocManaged(&c, N * sizeof(int));

    for (int i = 0; i < N; ++i) {
        a[i] = i;
        b[i] = 2 * i;
    }

    launch_add(a, b, c, N);
    cudaDeviceSynchronize();

    std::span<int> s(c, 10);
    for (auto [i, v] : std::views::enumerate(s)) {
        std::cout << i << ": " << v << "\n";
    }

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
}

