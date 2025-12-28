#pragma once

#include <cuda_runtime.h>
#include <cstdio>

constexpr int threads = 256;
constexpr int blocks = 256;

void launch_memory_add(const int* d_a, const int* d_b, int* d_c, int n);
void launch_compute_add(int repeats);
