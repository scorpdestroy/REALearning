#ifndef PRIME_COMMON_H
#define PRIME_COMMON_H

#include <cstdint>

#if defined(__CUDACC__)
    #define CUDA_HOST_DEV __host__ __device__
#else
    #define CUDA_HOST_DEV
#endif

inline CUDA_HOST_DEV bool is_prime(uint32_t n) {
    if (n < 2) return false;
    if (n == 2) return true;
    if (n % 2 == 0) return false;
    uint32_t d = 3;
    while ((uint64_t)d * d <= n) { 
        if (n % d == 0) return false;
        d += 2;
    }
    return true;
}

#endif
