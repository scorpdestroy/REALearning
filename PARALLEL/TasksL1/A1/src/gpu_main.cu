#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <random>
#include <cuda_runtime.h>

#include "prime_common.h"

struct GpuArgs {
    std::string dataset = "MIX";
    int64_t count = 100000;
    int block_size = 256;
    unsigned int seed = 42;
};

GpuArgs parse_arguments(int argc, char* argv[]) {
    GpuArgs args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--dataset" && i + 1 < argc) args.dataset = argv[++i];
        else if (arg == "--count" && i + 1 < argc) args.count = std::stoll(argv[++i]);
        else if (arg == "--block" && i + 1 < argc) args.block_size = std::stoi(argv[++i]);
        else if (arg == "--seed" && i + 1 < argc) args.seed = std::stoul(argv[++i]);
    }
    return args;
}

std::vector<uint32_t> generate_data(const std::string& type, int64_t count, unsigned int seed) {
    std::vector<uint32_t> data(count);
    std::mt19937 gen(seed);
    
    std::uniform_int_distribution<uint32_t> dist_mix(0, 4294967295U);
    std::uniform_int_distribution<uint32_t> dist_easy_even(2, 2000000000);
    std::uniform_int_distribution<uint32_t> dist_hard(3000000000, 4294967291U);

    for (int64_t i = 0; i < count; ++i) {
        if (type == "EASY") {
            uint32_t val = dist_easy_even(gen);
            data[i] = (val % 2 == 0) ? val : val + 1;
        } 
        else if (type == "HARD") {
            uint32_t val = dist_hard(gen);
            data[i] = (val % 2 != 0) ? val : val - 1;
        } 
        else {
            data[i] = dist_mix(gen);
        }
    }
    return data;
}

__global__ void prime_kernel(const uint32_t* values, uint8_t* flags, size_t count) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    flags[i] = is_prime(values[i]) ? 1 : 0;
}

int main(int argc, char* argv[]) {
    GpuArgs args = parse_arguments(argc, argv);

    std::vector<uint32_t> h_values = generate_data(args.dataset, args.count, args.seed);
    std::vector<uint8_t> h_flags(args.count, 0);

    uint32_t* d_values = nullptr;
    uint8_t* d_flags = nullptr;
    
    size_t values_size = args.count * sizeof(uint32_t);
    size_t flags_size = args.count * sizeof(uint8_t);

    auto e2e_start = std::chrono::high_resolution_clock::now();

    cudaMalloc(&d_values, values_size);
    cudaMalloc(&d_flags, flags_size);

    cudaMemcpy(d_values, h_values.data(), values_size, cudaMemcpyHostToDevice);

    int threadsPerBlock = args.block_size;
    int blocksPerGrid = (args.count + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    cudaEventRecord(start_event, 0);

    prime_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_values, d_flags, args.count);

    cudaEventRecord(stop_event, 0);
    
    cudaEventSynchronize(stop_event);

    float kernel_ms = 0;
    cudaEventElapsedTime(&kernel_ms, start_event, stop_event);

    cudaMemcpy(h_flags.data(), d_flags, flags_size, cudaMemcpyDeviceToHost);

    auto e2e_end = std::chrono::high_resolution_clock::now();
    double e2e_ms = std::chrono::duration<double, std::milli>(e2e_end - e2e_start).count();

    int64_t prime_count = 0;
    for (int64_t i = 0; i < args.count; ++i) {
        if (h_flags[i] == 1) prime_count++;
    }

    std::cout << "GPU_Local,cuda,kernel_and_e2e," << args.dataset << "," << args.count << ",," 
              << args.block_size << "," << args.seed << ",1," << e2e_ms << "," << kernel_ms 
              << "," << prime_count << ",1" << std::endl;

    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    cudaFree(d_values);
    cudaFree(d_flags);

    return 0;
}
