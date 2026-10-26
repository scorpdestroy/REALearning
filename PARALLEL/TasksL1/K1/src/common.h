
#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <iostream>

inline bool is_power_of_two(std::size_t n) {
    return n > 0 && (n & (n - 1)) == 0;
}

enum class InputType {
    Uniform,
    Sorted,
    Reverse,
    FewUnique,
    NearlySorted
};

inline const char* to_string(InputType t) {
    switch (t) {
        case InputType::Uniform:      return "uniform";
        case InputType::Sorted:       return "sorted";
        case InputType::Reverse:      return "reverse";
        case InputType::FewUnique:    return "fewunique";
        case InputType::NearlySorted: return "nearlysorted";
    }
    return "unknown";
}

inline InputType parse_input(const std::string& s) {
    if (s == "uniform")      return InputType::Uniform;
    if (s == "sorted")       return InputType::Sorted;
    if (s == "reverse")      return InputType::Reverse;
    if (s == "fewunique")    return InputType::FewUnique;
    if (s == "nearlysorted") return InputType::NearlySorted;
    throw std::runtime_error("unknown --input: " + s);
}

inline std::vector<std::uint32_t> generate_input(InputType type,
                                                 std::size_t n,
                                                 std::uint64_t seed,
                                                 double swap_ratio = 0.01)
{
    std::mt19937_64 rng(seed);
    std::vector<std::uint32_t> a(n);

    switch (type) {
        case InputType::Uniform: {
            std::uniform_int_distribution<std::uint32_t> dist(0u, 0xFFFFFFFFu);
            for (auto& x : a) x = dist(rng);
            break;
        }
        case InputType::Sorted: {
            for (std::size_t i = 0; i < n; ++i)
                a[i] = static_cast<std::uint32_t>(i);
            break;
        }
        case InputType::Reverse: {
            for (std::size_t i = 0; i < n; ++i)
                a[i] = static_cast<std::uint32_t>(n - 1 - i);
            break;
        }
        case InputType::FewUnique: {
            std::uniform_int_distribution<int> pick(0, 15);
            
            const std::uint32_t keys[16] = {
                0u, 17u, 91u, 123456u, 999999u, 1u<<20, 1u<<24,
                0x33333333u, 0x7FFFFFFFu, 0x80000000u, 0xAAAAAAAAu,
                0xCCCCCCCCu, 0xDEADBEEFu, 0xFFFFFFFEu, 0xFFFFFFFFu, 12345u
            };
            for (auto& x : a) x = keys[pick(rng)];
            break;
        }
        case InputType::NearlySorted: {
            for (std::size_t i = 0; i < n; ++i)
                a[i] = static_cast<std::uint32_t>(i);
            if (n < 2) break;
            std::size_t swaps = static_cast<std::size_t>(swap_ratio * n);
            if (swaps == 0) swaps = 1;
            std::uniform_int_distribution<std::size_t> pos(0, n - 1);
            for (std::size_t s = 0; s < swaps; ++s)
                std::swap(a[pos(rng)], a[pos(rng)]);
            break;
        }
    }
    return a;
}

inline std::vector<std::uint32_t> reference_sort(std::vector<std::uint32_t> v) {
    std::sort(v.begin(), v.end());
    return v;
}

inline bool arrays_equal(const std::vector<std::uint32_t>& a,
                         const std::vector<std::uint32_t>& b)
{
    return a.size() == b.size() &&
           std::memcmp(a.data(), b.data(), a.size() * sizeof(std::uint32_t)) == 0;
}

class Stopwatch {
public:
    void start() { t0_ = std::chrono::steady_clock::now(); }
    double stop_ms() const {
        auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0_).count();
    }
private:
    std::chrono::steady_clock::time_point t0_{};
};

struct Args {
    std::string mode   = "serial";   
    std::size_t n      = 1024;
    InputType   input  = InputType::Uniform;
    std::uint64_t seed = 2026;
    int         threads = 0;         
    int         block   = 256;       
    int         repeats = 1;
    double      swap_ratio = 0.01;   
};

inline Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc)
                throw std::runtime_error(std::string("missing value for ") + name);
            return argv[++i];
        };
        if      (k == "--mode")        a.mode    = need("--mode");
        else if (k == "--n")           a.n       = std::stoull(need("--n"));
        else if (k == "--input")       a.input   = parse_input(need("--input"));
        else if (k == "--seed")        a.seed    = std::stoull(need("--seed"));
        else if (k == "--threads")     a.threads = std::stoi(need("--threads"));
        else if (k == "--block")       a.block   = std::stoi(need("--block"));
        else if (k == "--repeats")     a.repeats = std::stoi(need("--repeats"));
        else if (k == "--swap-ratio")  a.swap_ratio = std::stod(need("--swap-ratio"));
        else if (k == "--help" || k == "-h") {
            std::cout <<
                "Usage: sort_cpu.exe --mode serial|omp|std --n N "
                "[--input uniform|sorted|reverse|fewunique|nearlysorted] "
                "[--seed S] [--threads P] [--repeats R] [--swap-ratio X]\n";
            std::exit(0);
        }
        else throw std::runtime_error("unknown arg: " + k);
    }
    return a;
}