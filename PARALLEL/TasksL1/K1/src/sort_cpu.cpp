
#include "common.h"

#include <cstdio>
#include <cstdlib>

#ifdef _OPENMP
#  include <omp.h>
#endif

static inline void compare_and_swap(std::uint32_t& x, std::uint32_t& y, bool ascending) {
    
    
    if (ascending) {
        if (x > y) std::swap(x, y);
    } else {
        if (x < y) std::swap(x, y);
    }
}

void bitonic_serial(std::vector<std::uint32_t>& a) {
    const std::size_t N = a.size();
    if (N < 2) return;
    if (!is_power_of_two(N))
        throw std::runtime_error("bitonic_serial: N must be a power of two");

    for (std::size_t k = 2; k <= N; k <<= 1) {
        for (std::size_t j = k >> 1; j > 0; j >>= 1) {
            for (std::size_t i = 0; i < N; ++i) {
                const std::size_t partner = i ^ j;
                if (i >= partner) continue;                 
                const bool ascending = (i & k) == 0;        
                compare_and_swap(a[i], a[partner], ascending);
            }
        }
    }
}

void bitonic_omp(std::vector<std::uint32_t>& a, int threads) {
    const std::size_t N = a.size();
    if (N < 2) return;
    if (!is_power_of_two(N))
        throw std::runtime_error("bitonic_omp: N must be a power of two");

#ifdef _OPENMP
    if (threads > 0) omp_set_num_threads(threads);
#endif

    std::uint32_t* data = a.data();

    for (std::size_t k = 2; k <= N; k <<= 1) {
        for (std::size_t j = k >> 1; j > 0; j >>= 1) {
            
            const std::int64_t Ni = static_cast<std::int64_t>(N);
            const std::size_t  jj = j;
            const std::size_t  kk = k;

            #pragma omp parallel for schedule(static)
            for (std::int64_t ii = 0; ii < Ni; ++ii) {
                const std::size_t i       = static_cast<std::size_t>(ii);
                const std::size_t partner = i ^ jj;
                if (i < partner) {
                    const bool ascending = (i & kk) == 0;
                    compare_and_swap(data[i], data[partner], ascending);
                }
            }
            
            
        }
    }
}

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);

        
        if (args.n == 0) {
            std::fprintf(stderr, "error: N=0 is not supported\n");
            return 2;
        }
        if (args.mode != "std" && !is_power_of_two(args.n)) {
            std::fprintf(stderr, "error: N=%zu is not a power of two (bitonic requires N=2^m)\n", args.n);
            return 2;
        }

        
        auto base = generate_input(args.input, args.n, args.seed, args.swap_ratio);
        auto expected = reference_sort(base);   

        double best_ms = 1e30;
        bool   all_correct = true;

        for (int r = 0; r < std::max(1, args.repeats); ++r) {
            auto work = base;                   
            Stopwatch sw;
            sw.start();

            if (args.mode == "serial") {
                bitonic_serial(work);
            } else if (args.mode == "std") {
                std::sort(work.begin(), work.end());
            } else if (args.mode == "omp") {
                bitonic_omp(work, args.threads);
            } else {
                std::fprintf(stderr, "error: unknown --mode %s\n", args.mode.c_str());
                return 2;
            }

            double ms = sw.stop_ms();
            best_ms = std::min(best_ms, ms);
            if (!arrays_equal(work, expected)) all_correct = false;
        }

        const char* algo = (args.mode == "serial") ? "bitonic_serial"
                         : (args.mode == "omp")    ? "bitonic_omp"
                         :                           "std_sort";
        const char* device = "cpu";

        for (int r = 0; r < std::max(1, args.repeats); ++r) {
            std::printf("local,%s,%s,%s,%zu,%d,,%llu,%d,%.6f,,,%d\n",
                        algo, device, to_string(args.input), args.n,
                        args.threads, (unsigned long long)args.seed, r,
                        best_ms, all_correct ? 1 : 0);
        }

        if (!all_correct) {
            std::fprintf(stderr, "CORRECTNESS FAILED\n");
            return 1;
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "exception: %s\n", e.what());
        return 3;   
    }
}