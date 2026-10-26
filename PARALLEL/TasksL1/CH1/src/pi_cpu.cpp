#include <iostream>
#include <string>
#include <chrono>
#include <cmath>
#include <random>
#include <omp.h>
#include <iomanip>

const double PI_REF = 3.14159265358979323846;
struct Args {
    std::string method = "";
    std::string mode = "";
    uint64_t n = 0;
    int threads = 1;
    unsigned int seed = 1234;
};
Args parse_arguments(int argc, char* argv[]) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--method" && i + 1 < argc) args.method = argv[++i];
        else if (arg == "--mode" && i + 1 < argc) args.mode = argv[++i];
        else if (arg == "--n" && i + 1 < argc) args.n = std::stoull(argv[++i]);
        else if (arg == "--threads" && i + 1 < argc) args.threads = std::stoi(argv[++i]);
        else if (arg == "--seed" && i + 1 < argc) args.seed = std::stoul(argv[++i]);
    }
    return args;
}
double run_midpoint_serial(uint64_t N) {
    double sum = 0.0;
    for (uint64_t i = 0; i < N; ++i) {
        double x = (static_cast<double>(i) + 0.5) / N;
        sum += 1.0 / (1.0 + x * x);
    }
    return 4.0 * sum / N;
}
double run_midpoint_omp(uint64_t N, int threads) {
    double sum = 0.0;
    omp_set_num_threads(threads);

    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (int64_t i = 0; i < static_cast<int64_t>(N); ++i) {
        double x = (static_cast<double>(i) + 0.5) / N;
        sum += 1.0 / (1.0 + x * x);
    }
    return 4.0 * sum / N;
}
double run_monte_carlo_serial(uint64_t N, unsigned int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    uint64_t points_inside = 0;
    for (uint64_t i = 0; i < N; ++i) {
        double x = dis(gen);
        double y = dis(gen);
        if (x * x + y * y <= 1.0) {
            points_inside++;
        }
    }
    return 4.0 * static_cast<double>(points_inside) / N;
}
double run_monte_carlo_omp(uint64_t N, int threads, unsigned int base_seed) {
    uint64_t total_points_inside = 0;
    omp_set_num_threads(threads);

    #pragma omp parallel reduction(+:total_points_inside)
    {
        int tid = omp_get_thread_num();
        std::mt19937 thread_gen(base_seed + tid);
        std::uniform_real_distribution<double> dis(0.0, 1.0);

        #pragma omp for schedule(static)
        for (int64_t i = 0; i < static_cast<int64_t>(N); ++i) {
            double x = dis(thread_gen);
            double y = dis(thread_gen);
            if (x * x + y * y <= 1.0) {
                total_points_inside++;
            }
        }
    }
    return 4.0 * static_cast<double>(total_points_inside) / N;
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
    if (args.mode != "serial" && args.mode != "omp") {
        std::cerr << "Error: Invalid mode. Use 'serial' or 'omp'." << std::endl;
        return 1;
    }

    double pi_calculated = 0.0;
    auto start_time = std::chrono::high_resolution_clock::now();

    if (args.method == "midpoint") {
        if (args.mode == "serial") {
            pi_calculated = run_midpoint_serial(args.n);
        } else {
            pi_calculated = run_midpoint_omp(args.n, args.threads);
        }
    } else if (args.method == "monte_carlo") {
        if (args.mode == "serial") {
            pi_calculated = run_monte_carlo_serial(args.n, args.seed);
        } else {
            pi_calculated = run_monte_carlo_omp(args.n, args.threads, args.seed);
        }
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    double abs_error = std::abs(pi_calculated - PI_REF);
    std::cout << args.method << ",CPU," << args.mode << "," << args.n << ","
    << (args.mode == "omp" ? std::to_string(args.threads) : "") << ",," 
    << (args.method == "monte_carlo" ? std::to_string(args.seed) : "") << ","
    << time_ms << ",," 
          << std::scientific << std::setprecision(15) << pi_calculated << "," 
          << std::setprecision(15) << abs_error << std::endl;


    return 0;
}
