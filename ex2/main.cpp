#include <cstdio>

void run_gpu_memory_test();
void run_gpu_register_tests();
void run_gpu_register_fma_tests();
void run_cpu_add_test();

int main()
{
    printf("Running GPU and CPU benchmarks\n");

    run_gpu_memory_test();
    run_gpu_register_tests();
    run_gpu_register_fma_tests();
    run_cpu_add_test();

    return 0;
}

