#pragma once

#include <random>
#include <vector>

class RandomManager {
private:
    uint64_t base_seed;
    std::mt19937_64 master_rng;

public:
    explicit RandomManager(uint64_t seed) : base_seed(seed), master_rng(seed) {}

    std::mt19937_64& get_master_rng() {
        return master_rng;
    }

    double get_uniform_double(double min_val, double max_val) {
        std::uniform_real_distribution<double> dist(min_val, max_val);
        return dist(master_rng);
    }

    int get_uniform_int(int min_idx, int max_idx) {
        std::uniform_int_distribution<int> dist(min_idx, max_idx);
        return dist(master_rng);
    }

    double get_normal_double(double mean, double sigma) {
        std::normal_distribution<double> dist(mean, sigma);
        return dist(master_rng);
    }

    std::vector<std::mt19937_64> create_thread_generators(int num_threads) {
        std::vector<std::mt19937_64> thread_rngs;
        thread_rngs.reserve(num_threads);
        
        for (int i = 0; i < num_threads; ++i) {
            uint64_t child_seed = master_rng(); 
            thread_rngs.emplace_back(child_seed);
        }
        return thread_rngs;
    }
};

struct GAParams {
    int P;        
    int D;        
    int G;        
    uint64_t seed;
};

double rastrigin_single(const double* x, int D);

double run_sequential_ga(const GAParams& params);

struct GATimingResult {
    double kernel_time_ms; 
    double total_time_ms;  
    double best_fitness;   
};

GATimingResult run_openmp_ga(const GAParams& params, int num_threads);