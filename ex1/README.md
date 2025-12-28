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

PS C:\> .\Windows\System32\nvidia-smi.exe
Sat Dec 27 17:10:16 2025
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 591.44                 Driver Version: 591.44         CUDA Version: 13.1     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5070      WDDM  |   00000000:01:00.0  On |                  N/A |
|  0%   34C    P8              8W /  250W |     749MiB /  12227MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            2308    C+G   ...IA App\CEF\NVIDIA Overlay.exe      N/A      |
|    0   N/A  N/A            2896    C+G   ...Chrome\Application\chrome.exe      N/A      |
|    0   N/A  N/A            3336    C+G   ...em32\ApplicationFrameHost.exe      N/A      |
|    0   N/A  N/A            8860    C+G   ...roadcast\NVIDIA Broadcast.exe      N/A      |
|    0   N/A  N/A            9888    C+G   ...y\StartMenuExperienceHost.exe      N/A      |
|    0   N/A  N/A           11400    C+G   ...2txyewy\CrossDeviceResume.exe      N/A      |
|    0   N/A  N/A           12880    C+G   ...Files\LM Studio\LM Studio.exe      N/A      |
|    0   N/A  N/A           12944    C+G   ...64__8wekyb3d8bbwe\Copilot.exe      N/A      |
|    0   N/A  N/A           13132    C+G   ...ms\Microsoft VS Code\Code.exe      N/A      |
|    0   N/A  N/A           13392    C+G   ...__kzh8wxbdkxb8p\DCv2\DCv2.exe      N/A      |
|    0   N/A  N/A           14980    C+G   ....0.3650.96\msedgewebview2.exe      N/A      |
|    0   N/A  N/A           15492    C+G   ...indows\System32\ShellHost.exe      N/A      |
|    0   N/A  N/A           16952    C+G   ...5n1h2txyewy\TextInputHost.exe      N/A      |
|    0   N/A  N/A           17532    C+G   ...pData\Local\Lark\app\Lark.exe      N/A      |
|    0   N/A  N/A           18140    C+G   C:\Windows\explorer.exe               N/A      |
|    0   N/A  N/A           18224    C+G   ...ntrolPanel\SystemSettings.exe      N/A      |
|    0   N/A  N/A           19944    C+G   ...8wekyb3d8bbwe\M365Copilot.exe      N/A      |
|    0   N/A  N/A           20540    C+G   ...t\Edge\Application\msedge.exe      N/A      |
|    0   N/A  N/A           20964      C   ...Files\LM Studio\LM Studio.exe      N/A      |
|    0   N/A  N/A           22420    C+G   ...yb3d8bbwe\WindowsTerminal.exe      N/A      |
|    0   N/A  N/A           22812    C+G   ...IA App\CEF\NVIDIA Overlay.exe      N/A      |
|    0   N/A  N/A           23020    C+G   ....0.3650.96\msedgewebview2.exe      N/A      |
|    0   N/A  N/A           23068    C+G   ....0.3650.96\msedgewebview2.exe      N/A      |
|    0   N/A  N/A           24412    C+G   ...8bbwe\PhoneExperienceHost.exe      N/A      |
|    0   N/A  N/A           25912    C+G   ...App_cw5n1h2txyewy\LockApp.exe      N/A      |
|    0   N/A  N/A           26576    C+G   ..._cw5n1h2txyewy\SearchHost.exe      N/A      |
|    0   N/A  N/A           27660    C+G   ...crosoft\OneDrive\OneDrive.exe      N/A      |
|    0   N/A  N/A           29036    C+G   ...Chrome\Application\chrome.exe      N/A      |
|    0   N/A  N/A           31188    C+G   ....0.3650.96\msedgewebview2.exe      N/A      |
+-----------------------------------------------------------------------------------------+
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
