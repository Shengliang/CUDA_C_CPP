# CUDA C++ Benchmark: Memory-Bound ADD Kernel

## Overview

This project benchmarks a **simple GPU kernel** that performs element-wise addition of two arrays. It measures:

* **Kernel execution time**
* **Compute throughput (Gops/s)**
* **Memory bandwidth (GB/s)**

It compares **measured performance** with **theoretical GPU peaks** for both **compute-bound** and **memory-bound workloads**.

---

## GPU Tested

* GPU: **NVIDIA RTX 5070**
* Driver Version: 591.44
* CUDA Version: 13.1
* Memory: 12 GB GDDR6
* Memory Bandwidth: ~900 GB/s (theoretical)
* FP32 Peak: 30.9 TFLOPS (FMA-based)
* FP32 ADD-only peak: ~14.3 TFLOPS
* Int32 ADD peak: ~14–15 TOPS

---

## Installation

1. **Install NVIDIA CUDA Toolkit 13.x**

   * [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)

2. **Install Visual Studio 2022 (x64)**

   * Include **Desktop development with C++** workload

3. **Install CMake 3.25+**

   * [https://cmake.org/download/](https://cmake.org/download/)

4. **Verify CUDA installation**:

```powershell
"C:\Windows\System32\nvidia-smi.exe"
```

---

## Compilation

Open **PowerShell (64-bit)** and run:

```powershell
cd <project_root>
.\build_and_run.ps1
```

* This script will:

  1. Create the `build` directory if it does not exist
  2. Configure the project with CMake
  3. Build the **Release configuration**
  4. Run the benchmark executable

---

## Running

If you prefer to run manually:

```powershell
cd build
cmake --build . --config Release
.\Release\cuda_cpp.exe
```

Example output:

```
Kernel time (per iteration): 0.337705 ms
Throughput: 49.68 Gops/s
Memory Bandwidth: 596.16 GB/s
Sample result c[0] = 0, c[N-1] = 50331645
Theoretical peak compute: 230000 Gops/s
Theoretical peak memory: 900 GB/s
```

---

## Interpreting Results

### Kernel Execution Time

* Average time per iteration, measured using **CUDA events**
* For small kernels, multiple iterations are averaged to avoid rounding errors

### Throughput (Gops/s)

* Number of operations per second:

```text
Throughput = N / kernel_time
```

* `N` = number of elements in the array

### Memory Bandwidth (GB/s)

* Measured as:

```text
Bandwidth = 3 * N * sizeof(int) / kernel_time
```

* 3 memory accesses per element: read `a`, read `b`, write `c`

---

### Theoretical vs Measured Performance

| Metric                         | Value (RTX 5070) | Notes                                        |
| ------------------------------ | ---------------- | -------------------------------------------- |
| FP32 FMA peak                  | 30.9 TFLOPS      | Compute-bound kernel                         |
| FP32 ADD-only peak             | ~14.3 TFLOPS     | Pure ADD instructions, no FMA                |
| Int32 ADD peak                 | ~14–15 TOPS      | Integer operations                           |
| Measured ADD kernel throughput | 49.7 Gops/s      | Memory-bound, saturates GPU memory bandwidth |
| Memory-bound peak (GB/s)       | 596 GB/s         | Matches measured memory throughput           |

**Explanation:**

* GPUs achieve peak FLOPs using **FMA instructions**, which count as **2 operations per cycle**
* Simple ADD-only kernels are **memory-bound**, and throughput is limited by memory bandwidth
* Your measured performance (~49.7 Gops/s) **matches memory-bound calculations**, not theoretical compute-bound peaks

---

## Notes

* Increasing **array size** (`N`) ensures stable timing
* Using **pinned host memory** (`cudaMallocHost`) improves H2D/D2H bandwidth
* **Multiple iterations** smooth out timing fluctuations for very fast kernels
* This benchmark is suitable for **memory-bound performance evaluation**
