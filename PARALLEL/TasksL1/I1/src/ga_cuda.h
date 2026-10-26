#pragma once
#include "ga_cpu.h"
#include <cuda_runtime.h>

// CUDA-ядро (объявлено глобально, чтобы main.cpp мог его вызвать для теста)
__global__ void evalRastriginKernel(const double* d_data, double* d_scores, int P, int D);

GATimingResult run_cuda_ga(const GAParams& params, int block_size = 256);