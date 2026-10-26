#include "ga_cuda.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono> // Добавлен для std::chrono

// Явное объявление внешней функции турнирного отбора, чтобы nvcc её видел
int tournament_selection(const std::vector<double>& scores, RandomManager& rng, int P);

// Константа Пи для использования внутри GPU ядра
__constant__ double d_PI = 3.14159265358979323846;

// Макрос для автоматической проверки ошибок API CUDA
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error в файле " << __FILE__ << " на строке " << __LINE__ \
                  << ": " << cudaGetErrorString(err) << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while(0)

/**
 * @brief CUDA-ядро расчета функции Растригина. Топология: один поток — одна особь.
 */
__global__ void evalRastriginKernel(const double* d_data, double* d_scores, int P, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= P) return; // Защита от выхода за границы популяции

    double sum = 0.0;
    int base_idx = i * D;

    for (int j = 0; j < D; ++j) {
        double val = d_data[base_idx + j];
        sum += val * val - 10.0 * cos(2.0 * d_PI * val);
    }

    d_scores[i] = 10.0 * D + sum;
}

/**
 * @brief Главный управляющий цикл хоста с переносом вычислений ядра на GPU
 */
GATimingResult run_cuda_ga(const GAParams& params, int block_size) {
    RandomManager rng(params.seed);

    // Выделение памяти на CPU
    std::vector<double> population(params.P * params.D);
    std::vector<double> next_population(params.P * params.D);
    std::vector<double> scores(params.P);

    // Инициализация популяции
    for (int i = 0; i < params.P * params.D; ++i) {
        population[i] = rng.get_uniform_double(-5.12, 5.12);
    }

    // --- ИНИЦИАЛИЗАЦИЯ И ВЫДЕЛЕНИЕ ПАМЯТИ НА GPU ---
    double* d_population = nullptr;
    double* d_scores = nullptr;
    size_t pop_size_bytes = params.P * params.D * sizeof(double);
    size_t scores_size_bytes = params.P * sizeof(double);

    CUDA_CHECK(cudaMalloc(&d_population, pop_size_bytes));
    CUDA_CHECK(cudaMalloc(&d_scores, scores_size_bytes));

    // Расчет конфигурации сетки
    int num_blocks = (params.P + block_size - 1) / block_size;

    double global_best_f = std::numeric_limits<double>::max();
    
    // Тайминги для замера полного времени
    auto total_start = std::chrono::high_resolution_clock::now();
    double cumulative_kernel_time_ms = 0.0;

    // Создание CUDA событий для точного замера времени работы ядра на GPU
    cudaEvent_t start_ev, stop_ev;
    CUDA_CHECK(cudaEventCreate(&start_ev));
    CUDA_CHECK(cudaEventCreate(&stop_ev));

    for (int generation = 0; generation < params.G; ++generation) {
        
        // 1. Копирование популяции Host -> Device по шине PCIe
        CUDA_CHECK(cudaMemcpy(d_population, population.data(), pop_size_bytes, cudaMemcpyHostToDevice));

        // 2. Асинхронный запуск ядра и замер его времени
        CUDA_CHECK(cudaEventRecord(start_ev, 0));
        evalRastriginKernel<<<num_blocks, block_size>>>(d_population, d_scores, params.P, params.D);
        CUDA_CHECK(cudaEventRecord(stop_ev, 0));
        
        // Синхронизация и проверка ошибок выполнения ядра
        CUDA_CHECK(cudaEventSynchronize(stop_ev));
        CUDA_CHECK(cudaGetLastError());

        float milliseconds = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_ev, stop_ev));
        cumulative_kernel_time_ms += milliseconds;

        // 3. Копирование результатов оценки Device -> Host
        CUDA_CHECK(cudaMemcpy(scores.data(), d_scores, scores_size_bytes, cudaMemcpyDeviceToHost));

        // Последовательный блок элитизма и операторов на CPU
        int elite_idx = 0;
        for (int i = 1; i < params.P; ++i) {
            if (scores[i] < scores[elite_idx]) elite_idx = i;
        }

        if (scores[elite_idx] < global_best_f) {
            global_best_f = scores[elite_idx];
        }

        std::copy(population.begin() + elite_idx * params.D,
                  population.begin() + (elite_idx + 1) * params.D,
                  next_population.begin());

        for (int i = 1; i < params.P; ++i) {
            int parent_a = tournament_selection(scores, rng, params.P);
            int parent_b = tournament_selection(scores, rng, params.P);
            double lambda = rng.get_uniform_double(0.0, 1.0);
            
            int child_base = i * params.D;
            int a_base = parent_a * params.D;
            int b_base = parent_b * params.D;

            for (int j = 0; j < params.D; ++j) {
                double child_gene = lambda * population[a_base + j] + (1.0 - lambda) * population[b_base + j];

                if (rng.get_uniform_double(0.0, 1.0) < (1.0 / params.D)) {
                    child_gene += rng.get_normal_double(0.0, 0.20);
                }

                if (child_gene < -5.12) child_gene = -5.12;
                if (child_gene > 5.12)  child_gene = 5.12;

                next_population[child_base + j] = child_gene;
            }
        }
        population = next_population;
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_time_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();

    // Освобождение GPU ресурсов
    CUDA_CHECK(cudaEventDestroy(start_ev));
    CUDA_CHECK(cudaEventDestroy(stop_ev));
    CUDA_CHECK(cudaFree(d_population));
    CUDA_CHECK(cudaFree(d_scores));

    GATimingResult result;
    result.kernel_time_ms = cumulative_kernel_time_ms;
    result.total_time_ms = total_time_ms;
    result.best_fitness = global_best_f;
    return result;
}
