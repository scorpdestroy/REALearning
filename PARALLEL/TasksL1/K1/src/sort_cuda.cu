
#include "common.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess) {                                           \
            std::fprintf(stderr, "CUDA error %s at %s:%d — %s\n",             \
                         cudaGetErrorName(err__), __FILE__, __LINE__,         \
                         cudaGetErrorString(err__));                          \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

__global__ void bitonic_step(std::uint32_t* a, unsigned N, unsigned j, unsigned k) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;                       
    const unsigned partner = i ^ j;
    if (i >= partner) return;                 

    const bool ascending = (i & k) == 0;      
    std::uint32_t x = a[i];
    std::uint32_t y = a[partner];
    if (ascending) {
        if (x > y) { a[i] = y; a[partner] = x; }
    } else {
        if (x < y) { a[i] = y; a[partner] = x; }
    }
}

struct GpuTimings {
    double alloc_ms    = 0.0;
    double transfer_ms = 0.0;
    double kernel_ms   = 0.0;
    double total_ms    = 0.0;
};

static bool gpu_sort(const std::vector<std::uint32_t>& input,
                     std::vector<std::uint32_t>&       output,
                     int                                block_size,
                     GpuTimings&                        t)
{
    const std::size_t N = input.size();
    const unsigned    Nu = static_cast<unsigned>(N);

    const unsigned grid = (Nu + block_size - 1) / block_size;

    
    cudaEvent_t e_total_start, e_total_stop;
    cudaEvent_t e_h2d_start,   e_h2d_stop;
    cudaEvent_t e_kern_start,  e_kern_stop;
    cudaEvent_t e_d2h_start,   e_d2h_stop;

    CUDA_CHECK(cudaEventCreate(&e_total_start));
    CUDA_CHECK(cudaEventCreate(&e_total_stop));
    CUDA_CHECK(cudaEventCreate(&e_h2d_start));
    CUDA_CHECK(cudaEventCreate(&e_h2d_stop));
    CUDA_CHECK(cudaEventCreate(&e_kern_start));
    CUDA_CHECK(cudaEventCreate(&e_kern_stop));
    CUDA_CHECK(cudaEventCreate(&e_d2h_start));
    CUDA_CHECK(cudaEventCreate(&e_d2h_stop));

    CUDA_CHECK(cudaEventRecord(e_total_start));

    
    Stopwatch sw_alloc; sw_alloc.start();
    std::uint32_t* d_a = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(std::uint32_t)));
    t.alloc_ms = sw_alloc.stop_ms();

    
    CUDA_CHECK(cudaEventRecord(e_h2d_start));
    CUDA_CHECK(cudaMemcpy(d_a, input.data(), N * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(e_h2d_stop));

    
    CUDA_CHECK(cudaEventRecord(e_kern_start));
    for (unsigned k = 2; k <= Nu; k <<= 1) {
        for (unsigned j = k >> 1; j > 0; j >>= 1) {
            bitonic_step<<<grid, block_size>>>(d_a, Nu, j, k);
            
            cudaError_t le = cudaGetLastError();
            if (le != cudaSuccess) {
                std::fprintf(stderr, "launch error at (k=%u, j=%u): %s\n",
                             k, j, cudaGetErrorString(le));
                std::exit(1);
            }
        }
    }
    
    CUDA_CHECK(cudaEventRecord(e_kern_stop));

    
    output.resize(N);
    CUDA_CHECK(cudaEventRecord(e_d2h_start));
    CUDA_CHECK(cudaMemcpy(output.data(), d_a, N * sizeof(std::uint32_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(e_d2h_stop));

    CUDA_CHECK(cudaEventRecord(e_total_stop));

    
    CUDA_CHECK(cudaEventSynchronize(e_total_stop));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e_h2d_start, e_h2d_stop));
    double h2d_ms = ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e_d2h_start, e_d2h_stop));
    double d2h_ms = ms;
    t.transfer_ms = h2d_ms + d2h_ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e_kern_start, e_kern_stop));
    t.kernel_ms = ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e_total_start, e_total_stop));
    t.total_ms = ms;

    
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_a));

    CUDA_CHECK(cudaEventDestroy(e_total_start));
    CUDA_CHECK(cudaEventDestroy(e_total_stop));
    CUDA_CHECK(cudaEventDestroy(e_h2d_start));
    CUDA_CHECK(cudaEventDestroy(e_h2d_stop));
    CUDA_CHECK(cudaEventDestroy(e_kern_start));
    CUDA_CHECK(cudaEventDestroy(e_kern_stop));
    CUDA_CHECK(cudaEventDestroy(e_d2h_start));
    CUDA_CHECK(cudaEventDestroy(e_d2h_stop));
    return true;
}

int main(int argc, char** argv) {
    try {
        Args a = parse_args(argc, argv);

        if (a.mode != "cuda") {
            std::fprintf(stderr, "error: this binary supports only --mode cuda\n");
            return 2;
        }
        if (a.n == 0) {
            std::fprintf(stderr, "error: N=0 is not supported\n");
            return 2;
        }
        if (!is_power_of_two(a.n)) {
            std::fprintf(stderr, "error: N=%zu is not a power of two\n", a.n);
            return 2;
        }
        if (a.block != 128 && a.block != 256 && a.block != 512) {
            std::fprintf(stderr, "error: --block must be 128, 256 or 512\n");
            return 2;
        }

        
        auto base     = generate_input(a.input, a.n, a.seed, a.swap_ratio);
        auto expected = reference_sort(base);

        int correct_all = 1;
        for (int r = 0; r < std::max(1, a.repeats); ++r) {
            std::vector<std::uint32_t> out;
            GpuTimings t;
            gpu_sort(base, out, a.block, t);

            const bool ok = arrays_equal(out, expected);
            if (!ok) correct_all = 0;

            std::printf("local,bitonic_cuda,gpu,%s,%zu,%d,%d,%llu,%d,%.6f,%.6f,%.6f,%d\n",
                        to_string(a.input), a.n, 0, a.block,
                        (unsigned long long)a.seed, r,
                        t.total_ms, t.kernel_ms, t.transfer_ms,
                        ok ? 1 : 0);
        }
        if (!correct_all) {
            std::fprintf(stderr, "CORRECTNESS FAILED\n");
            return 1;
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "exception: %s\n", e.what());
        return 3;
    }
}