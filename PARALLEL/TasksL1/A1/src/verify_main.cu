#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
#include <iomanip>
#include <cuda_runtime.h>
#include "prime_common.h"

__global__ void prime_kernel(const uint32_t* values, uint8_t* flags, size_t count) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    flags[i] = is_prime(values[i]) ? 1 : 0;
}

struct UnitTest {
    uint32_t input;
    bool expected;
    std::string reason;
};

bool run_unit_tests() {
    std::vector<UnitTest> tests = {
        {0,          false, "Граничный случай (0)"},
        {1,          false, "Граничный случай (1)"},
        {2,          true,  "Единственное четное простое (2)"},
        {3,          true,  "Малое простое (3)"},
        {5,          true,  "Малое простое (5)"},
        {97,         true,  "Малое простое (97)"},
        {4,          false, "Квадрат простого (4)"},
        {9,          false, "Квадрат простого (9)"},
        {25,         false, "Квадрат простого (25)"},
        {49,         false, "Квадрат простого (49)"},
        {15,         false, "Малый делитель (15)"},
        {21,         false, "Малый делитель (21)"},
        {27,         false, "Малый делитель (27)"},
        {65521,      true,  "Простое близко к 2^16"},
        {4294967291, true,  "Большое простое в uint32_t"},
        {4294967295, false, "Верхняя граница uint32_t (составное)"}
    };

    std::cout << "[INFO] Запуск обязательных точечных тестов (Раздел 5.1):\n";
    std::cout << "---------------------------------------------------------\n";
    bool all_passed = true;
    
    for (const auto& test : tests) {
        bool result = is_prime(test.input);
        bool passed = (result == test.expected);
        std::cout << "Вход: " << std::setw(11) << test.input 
                  << " | Ожидалось: " << (test.expected ? "Да" : "Нет")
                  << " | Получено: " << (result ? "Да" : "Нет")
                  << " | " << (passed ? "SUCCESS" : "FAIL") << " (" << test.reason << ")\n";
        if (!passed) all_passed = false;
    }
    std::cout << "---------------------------------------------------------\n";
    return all_passed;
}

bool verify_arrays(const uint32_t* values, const uint8_t* cpu_flags, const uint8_t* gpu_flags, size_t count) {
    size_t errors = 0;
    int64_t cpu_prime_count = 0;
    int64_t gpu_prime_count = 0;

    for (size_t i = 0; i < count; ++i) {
        if (cpu_flags[i] == 1) cpu_prime_count++;
        if (gpu_flags[i] == 1) gpu_prime_count++;
        if (cpu_flags[i] != gpu_flags[i]) {
            if (errors == 0) {
                std::cerr << "\n[ERROR] КРИТИЧЕСКОЕ РАСХОЖДЕНИЕ ДАННЫХ!\n";
                std::cerr << "Индекс расхождения: " << i << "\n";
                std::cerr << "Проверяемое число:  " << values[i] << "\n";
                std::cerr << "Флаг CPU (Serial):  " << (int)cpu_flags[i] << "\n";
                std::cerr << "Флаг GPU (CUDA):    " << (int)gpu_flags[i] << "\n";
            }
            errors++;
        }
    }

    std::cout << "\n[INFO] Результаты агрегации флагов:\n";
    std::cout << "Простых чисел на CPU: " << cpu_prime_count << "\n";
    std::cout << "Простых чисел на GPU: " << gpu_prime_count << "\n";

    if (cpu_prime_count != gpu_prime_count) {
        std::cerr << "[ERROR] Общее количество простых чисел НЕ совпадает!\n";
        return false;
    }

    if (errors > 0) {
        std::cerr << "[ERROR] Всего обнаружено байтовых расхождений: " << errors << "\n";
        return false;
    }

    std::cout << "[SUCCESS] Массивы CPU и GPU побитово и побайтово идентичны.\n";
    return true;
}

int main() {
    if (!run_unit_tests()) {
        std::cerr << "[FAIL] Точечные тесты провалены! Исправьте prime_common.h\n";
        return 1;
    }
    std::cout << "[PASS] Все точечные тесты успешно пройдены.\n\n";
    const size_t N = 500000;
    std::cout << "[INFO] Генерация стресс-теста на N = " << N << " элементов...\n";
    
    std::vector<uint32_t> h_values(N);
    std::vector<uint8_t> h_cpu_flags(N, 0);
    std::vector<uint8_t> h_gpu_flags(N, 0);
    for (size_t i = 0; i < N; ++i) {
        h_values[i] = (rand() << 16) | rand();
    }
    std::cout << "[INFO] Расчет на CPU (последовательный)... ";
    for (size_t i = 0; i < N; ++i) {
        h_cpu_flags[i] = is_prime(h_values[i]) ? 1 : 0;
    }
    std::cout << "Готово.\n";
    std::cout << "[INFO] Расчет на GPU (CUDA)... ";
    uint32_t* d_values = nullptr;
    uint8_t* d_flags = nullptr;
    
    cudaMalloc(&d_values, N * sizeof(uint32_t));
    cudaMalloc(&d_flags, N * sizeof(uint8_t));
    
    cudaMemcpy(d_values, h_values.data(), N * sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    int block = 256;
    int grid = (N + block - 1) / block;
    prime_kernel<<<grid, block>>>(d_values, d_flags, N);
    
    cudaDeviceSynchronize();
    cudaMemcpy(h_gpu_flags.data(), d_flags, N * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    std::cout << "Готово.\n";
    bool verified = verify_arrays(h_values.data(), h_cpu_flags.data(), h_gpu_flags.data(), N);
    cudaFree(d_values);
    cudaFree(d_flags);

    return verified ? 0 : 1;
}
