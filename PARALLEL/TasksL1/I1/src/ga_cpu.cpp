#include "ga_cpu.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <omp.h>
#include <chrono>


int tournament_selection(const std::vector<double>& scores, RandomManager& rng, int P) {
    int best_idx = rng.get_uniform_int(0, P - 1);
    
    for (int i = 1; i < 3; ++i) {
        int candidate_idx = rng.get_uniform_int(0, P - 1);
        if (scores[candidate_idx] < scores[best_idx]) { // Ловушка: ищем МИНИМУМ f
            best_idx = candidate_idx;
        }
    }
    return best_idx;
}

double run_sequential_ga(const GAParams& params) {
    RandomManager rng(params.seed);
    
    std::vector<double> population(params.P * params.D);
    std::vector<double> next_population(params.P * params.D);
    std::vector<double> scores(params.P);

    for (int i = 0; i < params.P * params.D; ++i) {
        population[i] = rng.get_uniform_double(-5.12, 5.12);
    }

    double global_best_f = std::numeric_limits<double>::max();

    for (int generation = 0; generation < params.G; ++generation) {
        for (int i = 0; i < params.P; ++i) {
            scores[i] = rastrigin_single(population.data() + i * params.D, params.D);
        }

        int elite_idx = 0;
        for (int i = 1; i < params.P; ++i) {
            if (scores[i] < scores[elite_idx]) {
                elite_idx = i;
            }
        }
        
        double current_best_f = scores[elite_idx];
        if (current_best_f < global_best_f) {
            global_best_f = current_best_f;
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

    return global_best_f;
}

int tournament_selection(const std::vector<double>& scores, RandomManager& rng, int P);

GATimingResult run_openmp_ga(const GAParams& params, int num_threads) {
    omp_set_num_threads(num_threads);

    RandomManager rng(params.seed);
    
    std::vector<double> population(params.P * params.D);
    std::vector<double> next_population(params.P * params.D);
    std::vector<double> scores(params.P);

    for (int i = 0; i < params.P * params.D; ++i) {
        population[i] = rng.get_uniform_double(-5.12, 5.12);
    }

    double global_best_f = std::numeric_limits<double>::max();
    
    double total_start = omp_get_wtime();
    double cumulative_kernel_time = 0.0;

    for (int generation = 0; generation < params.G; ++generation) {
        
        double kernel_start = omp_get_wtime();
        
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < params.P; ++i) {
            scores[i] = rastrigin_single(population.data() + i * params.D, params.D);
        }
        
        double kernel_end = omp_get_wtime();
        cumulative_kernel_time += (kernel_end - kernel_start);

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

    double total_end = omp_get_wtime();

    GATimingResult result;
    result.kernel_time_ms = cumulative_kernel_time * 1000.0;
    result.total_time_ms = (total_end - total_start) * 1000.0;
    result.best_fitness = global_best_f;
    return result;
}