

#include "gpu_memory_test.h"
#include "gpu_register_test.h"
#include "cpu_add_test.cpp"

int main(){
    int N_mem=64*1024*1024;
    int N_compute=64*1024*1024;

    printf("Running GPU and CPU benchmarks\n");

    run_memory_benchmark(N_mem);
    run_compute_benchmark(N_compute);
    run_cpu_benchmark(N_mem);

    return 0;
}
