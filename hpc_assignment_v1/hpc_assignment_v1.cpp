/*
 * Program Description:
 *
 * This program benchmarks and compares the performance of an element-wise square root kernel
 * executed using three different SYCL execution modes:
 *   1. CPU-only execution
 *   2. GPU-only execution
 *   3. Hybrid execution (splitting the workload between CPU and GPU)
 *
 * Objective:
 * - To evaluate the execution time and verify the numerical consistency of results across
 *   the three computational strategies.
 *
 * Methodology:
 * - The kernel computes the square root of `N` single-precision floating-point values
 *   (from 0.0 to N-1.0) and stores the results in a corresponding output vector.
 * - The program runs each configuration for `ITERATIONS` iterations to collect
 *   timing statistics. The first iteration is considered a warm-up run and is excluded
 *   from average timing calculations to avoid initialization overhead bias.
 * - Verification is performed by comparing all outputs from each method to ensure
 *   they are numerically equivalent within a small tolerance (ε = 1e-5).
 * - Optional detailed result output can be toggled using the `VERBOSE_OUTPUT` flag.
 *
 * Parameters:
 * - `N` (int): Total number of elements in the input/output vectors (default: 1000).
 * - `ITERATIONS` (int): Number of timing runs per method (default: 11).
 * - `VERBOSE_OUTPUT` (bool): Enables detailed printing of individual results if set to `true`.
 *
 * Requirements:
 * - A system with SYCL support, including both CPU and GPU devices.
 * - A DPC++/SYCL-compatible compiler (e.g., Intel oneAPI DPC++ or hipSYCL).
 *
 * Author: Ng Wei Jie
 * Date: 23 May 2025
 */



#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <numeric>

using namespace sycl;

// === Configuration Flags ===
constexpr int N =1000;
constexpr int ITERATIONS = 21;
const bool VERBOSE_OUTPUT = false;  // Set to true to enable detailed output

// === Benchmark Functions ===
float benchmark(queue& q, const std::vector<float>& input, std::vector<float>& output) {
    auto start = std::chrono::high_resolution_clock::now();
    {
        buffer<float> input_buf(input.data(), range<1>(N));
        buffer<float> output_buf(output.data(), range<1>(N));
        q.submit([&](handler& h) {
            auto in = input_buf.get_access<access::mode::read>(h);
            auto out = output_buf.get_access<access::mode::write>(h);
            h.parallel_for(range<1>(N), [=](id<1> i) {
                out[i] = sycl::sqrt(in[i]);
                });
            });
        q.wait();
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<float, std::milli>(end - start).count();
}

float benchmark_hybrid(queue& cpu_q, queue& gpu_q, const std::vector<float>& input, std::vector<float>& output) {
    float* usm_input = malloc_shared<float>(N, gpu_q);
    float* usm_output = malloc_shared<float>(N, gpu_q);

    std::copy(input.begin(), input.end(), usm_input);

    int mid = N / 2;

    auto start = std::chrono::high_resolution_clock::now();
    auto cpu_event = cpu_q.submit([&](handler& h) {
        h.parallel_for(range<1>(mid), [=](id<1> i) {
            usm_output[i] = sycl::sqrt(usm_input[i]);
            });
        });
    auto gpu_event = gpu_q.submit([&](handler& h) {
        h.parallel_for(range<1>(N - mid), [=](id<1> i) {
            int idx = mid + i;
            usm_output[idx] = sycl::sqrt(usm_input[idx]);
            });
        });

    cpu_event.wait();
    gpu_event.wait();
    auto end = std::chrono::high_resolution_clock::now();

    std::copy(usm_output, usm_output + N, output.begin());

    sycl::free(usm_input, gpu_q);
    sycl::free(usm_output, gpu_q);

    return std::chrono::duration<float, std::milli>(end - start).count();
}

int main() {
    std::cout << "-------------------------------------------------------\n";
    std::cout << " SYCL Element-wise Square Root Kernel Benchmark\n";
    std::cout << "-------------------------------------------------------\n";
    std::cout << " - Computes sqrt(x) for x = 0 ... N-1\n";
    std::cout << " - Execution modes: CPU-only | GPU-only | Hybrid\n";
    std::cout << " - Repeats: ITERATIONS = " << ITERATIONS
        << " (excluding 1st warm-up run)\n";
    std::cout << " - Vector size N = " << N << "\n";
    std::cout << " - VERBOSE_OUTPUT = "
        << (VERBOSE_OUTPUT ? "true" : "false")
        << " (set true for full result list)\n";
    std::cout << "-------------------------------------------------------\n";

    std::vector<float> input(N);
    std::iota(input.begin(), input.end(), 0.0f);

    std::vector<float> output_cpu(N), output_gpu(N), output_hybrid(N);

    try {
        queue cpu_q(cpu_selector_v);
        queue gpu_q(gpu_selector_v);

        std::cout << "[Device Info]\n";
        std::cout << "CPU: " << cpu_q.get_device().get_info<info::device::name>() << "\n";
        std::cout << "GPU: " << gpu_q.get_device().get_info<info::device::name>() << "\n";

        // Initial run for result checking
        benchmark(cpu_q, input, output_cpu);
        benchmark(gpu_q, input, output_gpu);
        benchmark_hybrid(cpu_q, gpu_q, input, output_hybrid);

        // Optional detailed result output
        if (VERBOSE_OUTPUT) {
            std::cout << "\n[Results Comparison - All Entries]\n";
            std::cout << std::fixed << std::setprecision(6);
            for (int i = 0; i < N; ++i) {
                std::cout << "sqrt(" << input[i] << ") = "
                    << "CPU: " << output_cpu[i]
                    << " | GPU: " << output_gpu[i]
                        << " | Hybrid: " << output_hybrid[i] << "\n";
            }
        }

        auto compare = [](float a, float b) { return std::fabs(a - b) < 1e-5f; };
        bool cpu_vs_gpu = std::equal(output_cpu.begin(), output_cpu.end(), output_gpu.begin(), compare);
        bool cpu_vs_hybrid = std::equal(output_cpu.begin(), output_cpu.end(), output_hybrid.begin(), compare);
        bool gpu_vs_hybrid = std::equal(output_gpu.begin(), output_gpu.end(), output_hybrid.begin(), compare);


        std::cout << "\n[Verification Results]\n";
        std::cout << "CPU vs GPU: " << (cpu_vs_gpu ? "Match" : "Mismatch") << "\n";
        std::cout << "CPU vs Hybrid: " << (cpu_vs_hybrid ? "Match" : "Mismatch") << "\n";
        std::cout << "GPU vs Hybrid: " << (gpu_vs_hybrid ? "Match" : "Mismatch") << "\n";
        

        // Timing
        std::vector<float> times_cpu, times_gpu, times_hybrid;

        std::cout << "\n[Execution Times in milliseconds for " << ITERATIONS << " iterations]\n";

        std::cout << "\nCPU-only times:\n";
        for (int i = 0; i < ITERATIONS; ++i) {
            float t = benchmark(cpu_q, input, output_cpu);
            times_cpu.push_back(t);
            std::cout << t << " ms\n";
        }

        std::cout << "\nGPU-only times:\n";
        for (int i = 0; i < ITERATIONS; ++i) {
            float t = benchmark(gpu_q, input, output_gpu);
            times_gpu.push_back(t);
            std::cout << t << " ms\n";
        }

        std::cout << "\nHybrid times:\n";
        for (int i = 0; i < ITERATIONS; ++i) {
            float t = benchmark_hybrid(cpu_q, gpu_q, input, output_hybrid);
            times_hybrid.push_back(t);
            std::cout << t << " ms\n";
        }

        auto average = [](const std::vector<float>& v) {
            return std::accumulate(v.begin() + 1, v.end(), 0.0f) / (v.size() - 1);
            };

        float avg_cpu = average(times_cpu);
        float avg_gpu = average(times_gpu);
        float avg_hybrid = average(times_hybrid);

        std::cout << "\n[Average Execution Time (Excluding First Run)]\n";
        std::cout << "CPU-only: " << avg_cpu << " ms\n";
        std::cout << "GPU-only: " << avg_gpu << " ms\n";
        std::cout << "Hybrid:   " << avg_hybrid << " ms\n";

        std::cout << "\n[Conclusion]\n";
        if (avg_hybrid < avg_gpu && avg_hybrid < avg_cpu)
            std::cout << "Hybrid is the fastest. It efficiently utilizes both CPU and GPU cores.\n";
        else if (avg_gpu < avg_cpu)
            std::cout << "GPU-only is the fastest. Best suited for highly parallel computation.\n";
        else
            std::cout << "CPU-only is the fastest. Possibly due to lower overhead or smaller workload.\n";

    }
    catch (const sycl::exception& e) {
        std::cerr << "SYCL Exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
