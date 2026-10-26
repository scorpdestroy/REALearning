#include <iostream>
#include <string>
#include <chrono>
#include <cmath>
#include <random>
#include <vector>
#include <cuda_runtime.h>
#include <iomanip>

const double PI_REF = 3.14159265358979323846;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

struct Args {
    std::string method = "";
    uint64_t n = 0;
    int block_size = 256;
    unsigned int seed = 1234;
};

Args parse_arguments(int argc, char* argv[]) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--method" && i + 1 < argc) args.method = argv[++i];
        else if (arg == "--n" && i + 1 < argc) args.n = std::stoull(argv[++i]);
        else if (arg == "--block" && i + 1 < argc) args.block_size = std::stoi(argv[++i]);
        else if (arg == "--seed" && i + 1 < argc) args.seed = std::stoul(argv[++i]);
    }
    return args;
}

__global__ void midpoint_kernel(uint64_t N, double* block_outputs) {
    extern __shared__ double cache[];

    int tid = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    double local_sum = 0.0;
    for (uint64_t i = global_id; i < N; i += stride) {
        double x = (static_cast<double>(i) + 0.5) / N;
        local_sum += 1.0 / (1.0 + x * x);
    }
    cache[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            cache[tid] += cache[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        block_outputs[blockIdx.x] = cache[0];
    }
}

__global__ void monte_carlo_kernel(uint64_t N, const double2* points, uint64_t* block_outputs) {
    extern __shared__ uint64_t mc_cache[];

    int tid = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    uint64_t local_count = 0;
    for (uint64_t i = global_id; i < N; i += stride) {
        double2 p = points[i];
        if (p.x * p.x + p.y * p.y <= 1.0) {
            local_count++;
        }
    }

    mc_cache[tid] = local_count;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            mc_cache[tid] += mc_cache[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_outputs[blockIdx.x] = mc_cache[0];
    }
}

int main(int argc, char* argv[]) {
    Args args = parse_arguments(argc, argv);
    if (args.n == 0) {
        std::cerr << "Error: N must be greater than 0." << std::endl;
        return 1;
    }
    if (args.method != "midpoint" && args.method != "monte_carlo") {
        std::cerr << "Error: Invalid method. Use 'midpoint' or 'monte_carlo'." << std::endl;
        return 1;
    }

    int block_size = args.block_size;
    int max_blocks = 512; 
    int grid_size = (args.n + block_size - 1) / block_size;
    if (grid_size > max_blocks) grid_size = max_blocks;

    double pi_calculated = 0.0;
    float kernel_ms = 0.0f;
    cudaEvent_t start_kernel, stop_kernel;
    CUDA_CHECK(cudaEventCreate(&start_kernel));
    CUDA_CHECK(cudaEventCreate(&stop_kernel));
    
    auto total_start = std::chrono::high_resolution_clock::now();

    if (args.method == "midpoint") {
        double* d_block_outputs = nullptr;
        std::vector<double> h_block_outputs(grid_size, 0.0);
        CUDA_CHECK(cudaMalloc(&d_block_outputs, grid_size * sizeof(double)));
        
        CUDA_CHECK(cudaEventRecord(start_kernel, 0));
        size_t shared_mem_size = block_size * sizeof(double);
        midpoint_kernel<<<grid_size, block_size, shared_mem_size>>>(args.n, d_block_outputs);
        CUDA_CHECK(cudaEventRecord(stop_kernel, 0));
        CUDA_CHECK(cudaEventSynchronize(stop_kernel));
        CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start_kernel, stop_kernel));
        
        CUDA_CHECK(cudaMemcpy(h_block_outputs.data(), d_block_outputs, grid_size * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_block_outputs));
        
        double total_sum = 0.0;
        for (int i = 0; i < grid_size; ++i) {
            total_sum += h_block_outputs[i];
        }
        pi_calculated = 4.0 * total_sum / args.n;

    } else if (args.method == "monte_carlo") {
        std::vector<double2> h_points(args.n);
        std::mt19937 gen(args.seed);
        std::uniform_real_distribution<double> dis(0.0, 1.0);
        for (uint64_t i = 0; i < args.n; ++i) {
            h_points[i] = make_double2(dis(gen), dis(gen));
        }
        
        double2* d_points = nullptr;
        uint64_t* d_block_outputs = nullptr;
        std::vector<uint64_t> h_block_outputs(grid_size, 0);

        CUDA_CHECK(cudaMalloc(&d_points, args.n * sizeof(double2)));
        CUDA_CHECK(cudaMalloc(&d_block_outputs, grid_size * sizeof(uint64_t)));

        CUDA_CHECK(cudaMemcpy(d_points, h_points.data(), args.n * sizeof(double2), cudaMemcpyHostToDevice));
        
        CUDA_CHECK(cudaEventRecord(start_kernel, 0));
        size_t shared_mem_size = block_size * sizeof(uint64_t);
        monte_carlo_kernel<<<grid_size, block_size, shared_mem_size>>>(args.n, d_points, d_block_outputs);
        CUDA_CHECK(cudaEventRecord(stop_kernel, 0));
        CUDA_CHECK(cudaEventSynchronize(stop_kernel));
        CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start_kernel, stop_kernel));
        
        CUDA_CHECK(cudaMemcpy(h_block_outputs.data(), d_block_outputs, grid_size * sizeof(uint64_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_points));
        CUDA_CHECK(cudaFree(d_block_outputs));
        
        uint64_t total_points_inside = 0;
        for (int i = 0; i < grid_size; ++i) {
            total_points_inside += h_block_outputs[i];
        }
        pi_calculated = 4.0 * static_cast<double>(total_points_inside) / args.n;
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_time_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();
    double abs_error = std::abs(pi_calculated - PI_REF);
    
    CUDA_CHECK(cudaEventDestroy(start_kernel));
    CUDA_CHECK(cudaEventDestroy(stop_kernel));

    // Корректный вывод для GPU: девайс=GPU, режим=cuda, потоков CPU нет (пропуск), блок_size=args.block_size
    std::cout << args.method << ",GPU,cuda," << args.n << ",," 
              << block_size << "," 
              << (args.method == "monte_carlo" ? std::to_string(args.seed) : "") << ","
              << total_time_ms << "," << kernel_ms << ","
              << std::scientific << std::setprecision(15) << pi_calculated << "," 
              << std::setprecision(15) << abs_error << std::endl;

    return 0;
}
