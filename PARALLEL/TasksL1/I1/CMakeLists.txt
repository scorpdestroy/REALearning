cmake_minimum_required(VERSION 3.18)

# ---------- 1. Проект ----------
project(ParallelGeneticAlgorithm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ---------- 2. Архитектура GPU ----------
# RTX 30xx -> 86, RTX 40xx -> 89, GTX 16xx -> 75, RTX 50xx -> 120
# Если не знаешь свою — оставь 86 (работает на большинстве карт).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

# ---------- 3. Явный поиск nvcc (чтобы не искал в E:\bin) ----------
if(NOT DEFINED CMAKE_CUDA_COMPILER)
    # Пробуем найти по стандартному пути установки CUDA
    file(GLOB _cuda_candidates
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*/bin/nvcc.exe"
    )
    if(_cuda_candidates)
        # Берём самый старший по имени (v12.6 > v12.4)
        list(SORT _cuda_candidates ORDER DESCENDING)
        list(GET _cuda_candidates 0 _cuda_nvcc)
        message(STATUS "Найден nvcc: ${_cuda_nvcc}")
        set(CMAKE_CUDA_COMPILER "${_cuda_nvcc}" CACHE FILEPATH "CUDA compiler" FORCE)
    else()
        message(WARNING
            "nvcc.exe не найден в 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*/bin/'. "
            "Укажи его вручную: cmake -DCMAKE_CUDA_COMPILER=<путь> ...")
    endif()
endif()

# ---------- 4. OpenMP ----------
if(MSVC)
    find_package(OpenMP REQUIRED)
endif()

# ---------- 5. Цель ----------
add_executable(parallel_ga
    src/main.cpp
    src/ga_cpu.cpp
    src/ga_cpu.h
    src/ga_cuda.cu
    src/ga_cuda.h
)

# Заголовочные файлы — чтобы main.cpp видел ga_cuda.h через cuda_runtime.h
target_include_directories(parallel_ga PRIVATE src)

# ---------- 6. Флаги C++ ----------
if(MSVC)
    # /openmp — OpenMP, /O2 — оптимизация, /EHsc — исключения
    target_compile_options(parallel_ga PRIVATE /openmp /O2 /EHsc)
else()
    target_compile_options(parallel_ga PRIVATE -O3 -fopenmp)
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(parallel_ga PRIVATE OpenMP::OpenMP_CXX)
endif()

# ---------- 7. Флаги CUDA ----------
target_compile_options(parallel_ga PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
)

# ---------- 8. Свойства CUDA ----------
set_target_properties(parallel_ga PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)