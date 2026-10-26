#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <iomanip>
#include <fstream>
#include <string>
#include <cstdlib>

#include "ga_cpu.h"
#include "ga_cuda.h"

constexpr double PI = 3.14159265358979323846;

double rastrigin_single(const double* x, int D) {
    double sum = 0.0;
    for (int i = 0; i < D; ++i) {
        sum += x[i] * x[i] - 10.0 * std::cos(2.0 * PI * x[i]);
    }
    return 10.0 * D + sum;
}

// ---------- Тесты ----------
struct TestCase {
    std::string name;
    std::vector<double> coords;
    double expected;
    std::string description;
};

void run_mathematical_tests() {
    std::cout << "==================================================\n";
    std::cout << " RUNNING AUTOMATED TESTS FOR RASTRIGIN FUNCTION\n";
    std::cout << "==================================================\n";

    std::vector<TestCase> tests = {
        {"Глобальный минимум", {0.0, 0.0}, 0.0, "f(0, ..., 0) = 0"},
        {"Смещение по одной оси", {1.0, 0.0}, 1.0, "cos(2*pi) = 1"},
        {"Целочисленные координаты", {1.0, 1.0}, 2.0, "Сумма квадратов целых координат"},
        {"Дробная координата (пик)", {0.5, 0.0}, 20.25, "cos(pi) = -1"}
    };

    int passed = 0;
    const double EPSILON = 1e-8;

    for (const auto& test : tests) {
        double actual = rastrigin_single(test.coords.data(), 2);
        double diff = std::abs(actual - test.expected);
        std::cout << "Тест: " << std::left << std::setw(30) << test.name << " -> ";
        if (diff < EPSILON) {
            std::cout << "[ PASSED ]\n";
            passed++;
        } else {
            std::cout << "[ FAILED ] (expected " << test.expected
                      << ", got " << actual << ", diff " << diff << ")\n";
        }
    }
    std::cout << "--------------------------------------------------\n";
    std::cout << "ИТОГ: Пройдено математических тестов " << passed << " из " << tests.size() << "\n";
    std::cout << "==================================================\n\n";
    assert(passed == tests.size() && "Критическая ошибка: Базовые математические тесты не пройдены!");
}

void test_random_reproducibility() {
    std::cout << "==================================================\n";
    std::cout << " RUNNING TESTS FOR RANDOM NUMBER GENERATION (RNG)\n";
    std::cout << "==================================================\n";

    uint64_t seed = 42;
    RandomManager rng1(seed), rng2(seed);

    double v1 = rng1.get_uniform_double(-5.12, 5.12);
    double v2 = rng2.get_uniform_double(-5.12, 5.12);
    int i1 = rng1.get_uniform_int(0, 100);
    int i2 = rng2.get_uniform_int(0, 100);
    double n1 = rng1.get_normal_double(0.0, 0.20);
    double n2 = rng2.get_normal_double(0.0, 0.20);

    std::cout << "Менеджер 1: Real=" << v1 << ", Int=" << i1 << ", Norm=" << n1 << "\n";
    std::cout << "Менеджер 2: Real=" << v2 << ", Int=" << i2 << ", Norm=" << n2 << "\n";

    bool ok = (v1 == v2) && (i1 == i2) && (n1 == n2);
    std::cout << (ok ? "[ PASSED ] Генерация строго воспроизводима при фиксированном seed.\n"
                     : "[ FAILED ] Нарушена детерминированность генератора!\n");
    std::cout << "==================================================\n\n";
    assert(ok);
}

// ---------- НОВЫЙ ТЕСТ: CPU vs GPU корректность (п. 4.4 методички) ----------
double test_cuda_vs_cpu() {
    std::cout << "==================================================\n";
    std::cout << " RUNNING TESTS FOR CUDA vs CPU CORRECTNESS\n";
    std::cout << "==================================================\n";

    const int P = 1000;
    const int D = 50;
    std::vector<double> population(P * D);
    std::vector<double> cpu_scores(P, 0.0);
    std::vector<double> gpu_scores(P, 0.0);

    RandomManager rng(777);
    for (int i = 0; i < P * D; ++i) {
        population[i] = rng.get_uniform_double(-5.12, 5.12);
    }

    // CPU reference
    for (int i = 0; i < P; ++i) {
        cpu_scores[i] = rastrigin_single(population.data() + i * D, D);
    }

    // GPU
    double *d_pop = nullptr, *d_scores = nullptr;
    cudaMalloc(&d_pop, P * D * sizeof(double));
    cudaMalloc(&d_scores, P * sizeof(double));
    cudaMemcpy(d_pop, population.data(), P * D * sizeof(double), cudaMemcpyHostToDevice);

    int block_size = 256;
    int num_blocks = (P + block_size - 1) / block_size;
    evalRastriginKernel<<<num_blocks, block_size>>>(d_pop, d_scores, P, D);
    cudaDeviceSynchronize();
    cudaMemcpy(gpu_scores.data(), d_scores, P * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_pop);
    cudaFree(d_scores);

    // Сравнение
    double max_diff = 0.0;
    for (int i = 0; i < P; ++i) {
        double diff = std::abs(cpu_scores[i] - gpu_scores[i]);
        if (diff > max_diff) max_diff = diff;
    }

    std::cout << "Максимальная разница CPU vs GPU: " << max_diff << "\n";
    if (max_diff < 1e-8) {
        std::cout << "[ PASSED ] CPU и GPU совпадают с точностью 1e-8.\n";
    } else {
        std::cout << "[ FAILED ] Расхождение превышает допуск!\n";
    }
    std::cout << "==================================================\n\n";
    assert(max_diff < 1e-8 && "CUDA core расходится с CPU");
    return max_diff;
}

// ---------- main ----------
int main() {
    run_mathematical_tests();
    test_random_reproducibility();
    test_cuda_vs_cpu();

    std::ofstream csv_file("results/experiment_results.csv");
    if (!csv_file.is_open()) {
        std::cerr << "Ошибка: Не удалось создать results/experiment_results.csv\n";
        return 1;
    }
    csv_file << "P,D,G,Mode,Threads,KernelTimeMs,TotalTimeMs,BestFitness\n";

    std::vector<int> pop_sizes = {4096, 16384, 32768};
    std::vector<int> dimensions = {1000};
    std::vector<int> cpu_threads = {1, 2, 4, 6, 8};
    int generations = 50;
    uint64_t base_seed = 12345;

    std::cout << "Запуск серии экспериментов для записи в CSV...\n";

    for (int P : pop_sizes) {
        for (int D : dimensions) {
            GAParams params{P, D, generations, base_seed};
            std::cout << "Тестирование: P = " << P << ", D = " << D << std::endl;

            for (int threads : cpu_threads) {
                GATimingResult res = run_openmp_ga(params, threads);
                csv_file << P << "," << D << "," << generations << ","
                         << "OpenMP," << threads << ","
                         << res.kernel_time_ms << "," << res.total_time_ms << ","
                         << res.best_fitness << "\n";
            }

            GATimingResult res_cuda = run_cuda_ga(params, 256);
            csv_file << P << "," << D << "," << generations << ","
                     << "CUDA,256,"
                     << res_cuda.kernel_time_ms << "," << res_cuda.total_time_ms << ","
                     << res_cuda.best_fitness << "\n";
        }
    }

    csv_file.close();
    std::cout << "Эксперименты завершены. Данные сохранены в 'results/experiment_results.csv'\n";
    return 0;
}