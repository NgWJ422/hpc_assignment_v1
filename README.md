# Heterogeneous Execution with SYCL on CPU and GPU

**Group Members**:

* Ng Wei Jie
* Wee Mao Phin
* Lee Zhao Tian

**Course**: BERR4223 – High Performance Computing
**Assignment Topic**: Heterogeneous Execution with SYCL

---

## Overview

This project is part of our coursework for the *High Performance Computing* module. The objective is to implement and benchmark a parallel computation task using **SYCL 2020** and **Intel oneAPI**, targeting heterogeneous execution across CPU and GPU devices. The kernel selected for this project performs an **element-wise arithmetic square root operation**.

---

## Assignment Objectives

The specific goals of this assignment are as follows:

* **Kernel Selection**: Implement an arithmetic element-wise square root operation.
* **Execution Modes**:

  * **CPU-only execution**: Utilizing `cpu_selector_v`.
  * **GPU-only execution**: Utilizing `gpu_selector_v` on Intel oneAPI-compatible GPUs.
  * **Hybrid execution**: Splitting the workload between CPU and GPU using Unified Shared Memory (USM) and concurrent SYCL queues.
* **Performance Evaluation**: Measure and compare execution time across different strategies.
* **Numerical Verification**: Ensure the computational outputs are consistent and accurate within a small floating-point tolerance.

---

## Program Description

This program benchmarks and compares the performance of an element-wise square root kernel under three SYCL execution paradigms:

1. **CPU-only Execution**
2. **GPU-only Execution**
3. **Hybrid Execution (CPU + GPU Split)**

---

## Methodology

* The kernel processes `N` single-precision floating-point numbers ranging from `0.0` to `N-1.0`, storing the square root of each element in an output vector.
* Each execution mode is benchmarked over `ITERATIONS` cycles. The first iteration is treated as a warm-up and excluded from average timing calculations to minimize initialization overhead.
* Hybrid mode divides the input dataset evenly between CPU and GPU using USM and synchronizes kernel execution via `wait()`.
* Output vectors from all configurations are cross-validated for numerical consistency within an absolute tolerance of ε = 1e-5.
* Optional detailed output of computed results is controlled using the `VERBOSE_OUTPUT` flag.

---

## Parameters

| Parameter        | Type | Description                                           | Default Value |
| ---------------- | ---- | ----------------------------------------------------- | ------------- |
| `N`              | int  | Number of elements in input/output vectors            | 1000          |
| `ITERATIONS`     | int  | Number of execution cycles per mode                   | 11            |
| `VERBOSE_OUTPUT` | bool | Enables detailed output of results when set to `true` | false         |

---

## Requirements

* A system with SYCL support for both CPU and GPU devices.
* A DPC++/SYCL-compatible compiler, such as:

  * **Intel oneAPI DPC++**
  * **hipSYCL**
 
---

## Conclusion

This project demonstrates the benefits and trade-offs of heterogeneous computing using SYCL. By comparing CPU, GPU, and hybrid execution strategies, we gain insights into performance optimization and device utilization in modern high-performance computing environments.

---

## Why first iteration is slower:

The significantly longer execution time for the **first iteration** on both **CPU** and **GPU** is a well-known phenomenon in heterogeneous programming frameworks like SYCL, CUDA, and OpenCL. This behavior can be attributed to **initialization overhead**. Here's a breakdown of why it happens:

### 🔧 1. **SYCL Runtime Initialization**

* On the **first use** of a SYCL queue, the SYCL runtime initializes:

  * The selected device backend (e.g., OpenCL, Level Zero)
  * The device context and command queues
  * The JIT compiler (if required)
* This setup is **amortized** over subsequent kernel launches, so it's only noticeable on the **first invocation**.

### 🚀 2. **Kernel Compilation (JIT or Ahead-of-Time)**

* If your SYCL implementation uses **Just-In-Time (JIT) compilation**, then:

  * The kernel must be compiled at runtime before it can be executed on the device.
  * This includes translating from SPIR-V or intermediate code to native ISA for the target CPU or GPU.

### 📦 3. **Memory Allocation and USM Initialization**

* The first call to `malloc_shared`, `malloc_device`, or `malloc_host` may involve internal memory pool initialization, registration with the device driver, or page mapping operations.

### 🔁 4. **Driver-Level Warmup**

* On the GPU, driver-level warmup may include:

  * Shader pipeline initialization
  * Power management state transitions (from idle to active)
  * Caching structures being populated


### ⏱ Why Later Iterations Are Faster

Once the initial setup is complete, subsequent kernel launches:

* Reuse existing device queues and memory pools
* Use already-compiled kernels (or cached binaries)
* Avoid the expensive setup path

---

### 📉 Practical Tip

When benchmarking, it is common to:

* **Discard the first iteration** to avoid skewed results
* **Warm up the device** with a dummy kernel before actual measurement

For example:

```cpp
// Optional warm-up before timing loop
queue.submit([&](sycl::handler& h) {
    h.single_task([=]() {});
}).wait();
```

### 🧠 Summary

| Reason                    | Description                                         |
| ------------------------- | --------------------------------------------------- |
| SYCL runtime init         | Sets up backend, queues, and context                |
| Kernel JIT compilation    | Translates SYCL code to device-specific ISA         |
| USM and memory pool setup | Allocates shared/device memory, maps pages          |
| GPU driver warmup         | Loads microcode, power scaling, command queue setup |

This overhead is **normal** and does not indicate a problem in your code.


---

## Tolerance in floating point comparison

Yes, the tolerance in floating-point comparisons between CPU and GPU results is **indeed due to differences in how floating-point arithmetic is implemented on different hardware architectures**. Let's delve into the technical details:


### ⚙️ **Why Use a Tolerance in Floating-Point Comparisons?**

Floating-point numbers (like `float` and `double`) are represented using finite binary approximations of real numbers. Arithmetic operations on these approximations can introduce small rounding errors.

These discrepancies become especially apparent when comparing results computed on different hardware (e.g., CPU vs. GPU), due to the **variability in the following factors**:


### 🔍 **Key Architectural and Implementation Differences**

#### 1. **Instruction Sets and Optimizations**

* **CPU** uses instruction sets like **x86-64 (Intel/AMD)** with **SSE/AVX** for SIMD operations.
* **GPU** uses **SIMT (Single Instruction, Multiple Threads)** execution models and has its own set of **floating-point units (FPUs)**.
* The sequence of operations and optimizations (e.g., fused multiply-add) may differ, even if the high-level operation is the same.

#### 2. **Floating-Point Precision and Compliance**

* Some GPUs, especially older or low-power ones, may **not strictly comply with IEEE 754** (the standard for floating-point arithmetic) in the same way CPUs do.
* For instance, GPUs may:

  * Use **fused multiply-add (FMA)** by default, which can slightly alter rounding.
  * Truncate intermediate results differently.
  * Skip some denormal number handling for performance.

#### 3. **Order of Operations**

* Due to **parallel execution and reduction patterns**, the **order of operations can vary**, and since floating-point addition is **not associative**, small errors can accumulate differently.

#### 4. **Precision and Rounding Modes**

* Rounding behavior (toward zero, nearest, etc.) may vary subtly between devices unless explicitly controlled.
* Intermediate values in GPU may be stored in **lower or higher precision** (e.g., single vs. half precision).


### 🧪 Example:

```cpp
float a = 0.1f + 0.2f;  // May not be exactly 0.3f due to binary approximation
```

This trivial example can already show subtle differences between CPU and GPU depending on how the instructions are executed.


### ✅ Best Practice

To compare floating-point values:

```cpp
std::fabs(a - b) < epsilon
```

Where `epsilon` is a small value like `1e-5f` for `float`, or `1e-9` for `double`.

This is essential in heterogeneous computing (CPU + GPU + FPGA), where exact bitwise equality is neither expected nor required in many numerical applications.


### Summary

| Factor              | CPU                               | GPU                            |
| ------------------- | --------------------------------- | ------------------------------ |
| Instruction Set     | x86 / AVX                         | CUDA Cores / SIMT              |
| IEEE 754 Compliance | Strict (usually full compliance)  | May be relaxed for performance |
| Precision Control   | High (configurable rounding, FPU) | Often fixed or optimized       |
| Intermediate Values | Full 32/64-bit                    | Might use lower (e.g., 16-bit) |
| Operation Order     | Sequential or parallel (SIMD)     | Massively parallel (SIMT)      |

