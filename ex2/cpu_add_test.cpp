

#include <vector>
#include <chrono>
#include <cstdio>

void run_cpu_benchmark(int N){
    std::vector<int> a(N,1), b(N,2), c(N,0);

    auto start = std::chrono::high_resolution_clock::now();
    for(int i=0;i<N;i++){
        c[i] = a[i]+b[i];
    }
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double,std::milli>(end-start).count();
    double gops = double(N)/(ms/1000.0)/1e9;

    printf("\n--- CPU ADD Benchmark ---\n");
    printf("Kernel time (ms): %7.3f, Throughput: %7.3f Gops/s\n", ms, gops);
    printf("Sample result c[0]=%d, c[N-1]=%d\n", c[0], c[N-1]);
}
