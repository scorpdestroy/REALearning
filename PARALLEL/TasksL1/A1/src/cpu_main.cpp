#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <random>
#include <omp.h>
#include "prime_common.h"

struct AppArgs {
    std::string mode = "serial";
    std::string dataset = "MIX";
    int64_t count = 100000;
    int threads = 4;
    unsigned int seed = 42;
    std::string schedule_type = "static";
    int chunk_size = 0;
};

AppArgs parse_arguments(int argc, char* argv[]) {
    AppArgs args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--mode" && i + 1 < argc) args.mode = argv[++i];
        else if (arg == "--dataset" && i + 1 < argc) args.dataset = argv[++i];
        else if (arg == "--count" && i + 1 < argc) args.count = std::stoll(argv[++i]);
        else if (arg == "--threads" && i + 1 < argc) args.threads = std::stoi(argv[++i]);
        else if (arg == "--seed" && i + 1 < argc) args.seed = std::stoul(argv[++i]);
        else if (arg == "--schedule" && i + 1 < argc) args.schedule_type = argv[++i];
        else if (arg == "--chunk" && i + 1 < argc) args.chunk_size = std::stoi(argv[++i]);
    }
    return args;
}

std::vector<uint32_t> generate_data(const std::string& type, int64_t count, unsigned int seed) {
    std::vector<uint32_t> data(count);
    std::mt19937 gen(seed);
    
    std::uniform_int_distribution<uint32_t> dist_mix(0, 4294967295UI64);
    std::uniform_int_distribution<uint32_t> dist_easy_even(2, 2000000000);
    std::uniform_int_distribution<uint32_t> dist_hard(3000000000, 4294967291UI64);

    for (int64_t i = 0; i < count; ++i) {
        if (type == "EASY") {
            uint32_t val = dist_easy_even(gen);
            data[i] = (val % 2 == 0) ? val : val + 1;
        } 
        else if (type == "HARD") {
            // Большие нечетные числа
            uint32_t val = dist_hard(gen);
            data[i] = (val % 2 != 0) ? val : val - 1;
        } 
        else { // MIX
            data[i] = dist_mix(gen);
        }
    }
    return data;
}

int main(int argc, char* argv[]) {
    AppArgs args = parse_arguments(argc, argv);

    std::vector<uint32_t> values = generate_data(args.dataset, args.count, args.seed);
    std::vector<uint8_t> flags(args.count, 0);

    double wall_ms = 0.0;
    int64_t prime_count = 0;

    if (args.mode == "serial") {
        auto start = std::chrono::high_resolution_clock::now();

        for (int64_t i = 0; i < args.count; ++i) {
            flags[i] = is_prime(values[i]) ? 1 : 0;
        }

        auto end = std::chrono::high_resolution_clock::now();
        wall_ms = std::chrono::duration<double, std::milli>(end - start).count();
    } 
    else if (args.mode == "omp") {
        omp_set_num_threads(args.threads);
        
        auto start = std::chrono::high_resolution_clock::now();

        if (args.schedule_type == "dynamic") {
            if (args.chunk_size > 0) {
                #pragma omp parallel for schedule(dynamic, args.chunk_size)
                for (int64_t i = 0; i < args.count; ++i) {
                    flags[i] = is_prime(values[i]) ? 1 : 0;
                }
            } else {
                #pragma omp parallel for schedule(dynamic)
                for (int64_t i = 0; i < args.count; ++i) {
                    flags[i] = is_prime(values[i]) ? 1 : 0;
                }
            }
        } 
        else { // По умолчанию или явный static
            #pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < args.count; ++i) {
                flags[i] = is_prime(values[i]) ? 1 : 0;
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        wall_ms = std::chrono::duration<double, std::milli>(end - start).count();
    } 
    else {
        std::cerr << "Unknown mode: " << args.mode << std::endl;
        return 1;
    }

    for (int64_t i = 0; i < args.count; ++i) {
        if (flags[i] == 1) {
            prime_count++;
        }
    }

    std::cout << "CPU_Local," << args.mode << "," << args.schedule_type << "," 
              << args.dataset << "," << args.count << "," << args.threads << ",," 
              << args.seed << ",1," << wall_ms << ",," << prime_count << ",1" << std::endl;

    return 0;
}
