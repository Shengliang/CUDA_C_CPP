# CUDA C++ Project (Windows 11, CMake, MSVC)

This project demonstrates how to build and run a **CUDA C++ application** on **Windows 11** using **CMake**, **MSVC (Visual Studio Build Tools)**, and the **NVIDIA CUDA Toolkit**.

The setup supports:

* **C++23 for host code**
* **C++20 for CUDA (`.cu`) code** (maximum supported by NVCC)

---

## 1. Prerequisites

### 1.1 Hardware

* NVIDIA GPU with CUDA support
  (e.g. RTX 40xx / 50xx series)

Verify:

```powershell
nvidia-smi
```

---

### 1.2 Windows

* Windows 11 (64-bit)

---

### 1.3 Visual Studio Build Tools (MSVC)

Install **Build Tools for Visual Studio 2022**:

1. Download from:
   [https://visualstudio.microsoft.com/visual-cpp-build-tools/](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. During installation, select:

   * **Desktop development with C++**
3. Ensure the following components are included:

   * MSVC v143 toolset
   * Windows 10/11 SDK

Verify:

```powershell
cl
```

(Use **Developer Command Prompt for VS 2022** if needed.)

---

### 1.4 CUDA Toolkit

Install the NVIDIA CUDA Toolkit:

1. Download from:
   [https://developer.nvidia.com/cuda-toolkit](https://developer.nvidia.com/cuda-toolkit)
2. Install **CUDA Toolkit** (default options are fine)
3. Ensure CUDA is added to PATH

Verify:

```powershell
nvcc --version
```

---

### 1.5 CMake

Install CMake (3.26+ recommended):

* Download: [https://cmake.org/download/](https://cmake.org/download/)
* During installation, check:
  **“Add CMake to system PATH”**

Verify:

```powershell
cmake --version
```

---

## 2. Project Structure

```
ex1/
├── CMakeLists.txt
├── main.cpp        # Host code (C++23)
├── kernels.cu      # CUDA kernels (C++20)
└── build/          # Generated build directory
```

---

## 3. Build Instructions

All commands below are run from **PowerShell** or **VS Code terminal**.

### 3.1 Create build directory

From the project root:

```powershell
mkdir build
cd build
```

---

### 3.2 Configure with CMake

```powershell
cmake ..
```

Expected output includes:

* MSVC compiler detected
* CUDA compiler detected
* No configuration errors

---

### 3.3 Compile

```powershell
cmake --build . --config Release
```

This produces the executable:

```
build\Release\cuda_cpp.exe
```

---

## 4. Run Instructions

### 4.1 Run from PowerShell

From the `build` directory:

```powershell
.\Release\cuda_cpp.exe
```

Example output:

```
0: 0
1: 3
2: 6
3: 9
...
```

---

### 4.2 Debugging CUDA runtime issues (optional)

Force synchronous CUDA execution to surface errors:

```powershell
set CUDA_LAUNCH_BLOCKING=1
.\Release\cuda_cpp.exe
```

---

## 5. Common Issues & Solutions

### Issue: `cmake` not found

* CMake not installed or not in PATH
* Reinstall CMake and enable “Add to PATH”

---

### Issue: `nvcc` not found

* CUDA Toolkit not installed or PATH not set
* Reinstall CUDA Toolkit

---

### Issue: C++23 features fail in `.cu` files

* NVCC does **not** support C++23
* Use C++23 only in `.cpp` files
* Keep `.cu` files at C++20 or lower

---

### Issue: Executable not found

* You must run from `Release\`:

```powershell
.\Release\cuda_cpp.exe
```

---

## 6. Notes on C++ Standards

| File Type      | Compiler    | Standard        |
| -------------- | ----------- | --------------- |
| `.cpp`         | MSVC        | C++23           |
| `.cu` (host)   | NVCC + MSVC | C++20           |
| `.cu` (device) | NVCC        | C++17/20 subset |

This separation is **required** for stable CUDA builds on Windows.

---

## 7. Next Steps (Optional)

* Add CUDA performance benchmarks (bandwidth / FLOPS)
* Enable CUDA streams and async memory
* Profile with **Nsight Compute** or **Nsight Systems**
* Open the generated `.sln` file in Visual Studio

---

## 8. Clean Rebuild

If anything goes wrong:

```powershell
cd build
rm -r * -Force
cmake ..
cmake --build . --config Release
```

---

## Summary

```powershell
# Configure
cmake ..

# Build
cmake --build . --config Release

# Run
.\Release\cuda_cpp.exe
```

This is the **correct, production-grade workflow** for CUDA C++ on Windows 11.
