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

## Execution Instructions

1. Ensure your system is configured with the necessary SYCL and Intel oneAPI environment.
2. Compile the program using a compatible SYCL compiler.
3. Run the program and observe the benchmark results and verification output.

---

## Conclusion

This project demonstrates the benefits and trade-offs of heterogeneous computing using SYCL. By comparing CPU, GPU, and hybrid execution strategies, we gain insights into performance optimization and device utilization in modern high-performance computing environments.

---
