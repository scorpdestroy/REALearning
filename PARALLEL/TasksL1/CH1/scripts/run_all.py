import subprocess
import sys
import os
import math
import random
import csv

# ======================================================================
# НАСТРОЙКИ И КОНСТАНТЫ
# ======================================================================
PI_REF = 3.14159265358979323846

# Параметры массового эксперимента (Пункт 5.1 методички)
N_LIST = [10**3, 10**4, 10**5, 10**6, 10**7]
CPU_THREADS = [1, 2, 4, 6, 8, 12]
GPU_BLOCK_SIZES = [128, 256, 512]
NUM_REPEATS = 10
WARMUP_REPEATS = 2

# Автоматически фильтруем потоки под вашу систему
max_system_threads = os.cpu_count() or 1
cpu_threads_filtered = [t for t in CPU_THREADS if t <= max_system_threads]

os.makedirs("data", exist_ok=True)
csv_filepath = os.path.join("data", "measurements.csv")

# ======================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ======================================================================
def run_process(cmd):
    """Запускает процесс и возвращает код возврата, stdout и stderr"""
    try:
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding='utf-8')
        return res.returncode, res.stdout.strip(), res.stderr.strip()
    except Exception as e:
        return -1, "", str(e)

def print_result(test_name, is_ok, message=""):
    status = "[\033[92mУСПЕШНО\033[0m]" if is_ok else "[\033[91mОШИБКА\033[0m]"
    print(f"{status} {test_name} {message}")

def execute_and_save(cmd, is_warmup, repeat_idx, seeds_list, r_idx, method):
    """Выполняет команду массового эксперимента и сохраняет результат в CSV"""
    if is_warmup:
        run_process(cmd)
        return
        
    code, out, err = run_process(cmd)
    if code != 0 or not out:
        return

    parts = out.split(",")
    # Вставляем repeat на нужную позицию (Пункт 5.2 методички)
    parts.insert(7, str(repeat_idx))
    parts.append("Local_PC")

    with open(csv_filepath, mode="a", newline="", encoding="utf-8") as f:
        f.write(",".join(parts) + "\n")

# ======================================================================
# ЭТАП 1: ВАЛИДАЦИЯ И ГРАНИЧНЫЕ УСЛОВИЯ (РАЗДЕЛ 6)
# ======================================================================
print("=== ЭТАП 1: ЗАПУСК ПРОВЕРКИ ГРАНИЧНЫХ УСЛОВИЙ И КОРРЕКТНОСТИ ===\n")

if not os.path.exists("pi_cpu.exe") or not os.path.exists("pi_gpu.exe"):
    print("[\033[91mКРИТИЧЕСКАЯ ОШИБКА\033[0m] Сначала скомпилируйте pi_cpu.exe и pi_gpu.exe в корне проекта!")
    sys.exit(1)

# Тест 1: Проверка N = 0
print("--- Тест 1: Проверка N = 0 ---")
_, out, err = run_process("pi_cpu.exe --method midpoint --mode serial --n 0")
is_cpu_0_ok = "error" in out.lower() or "error" in err.lower() or _ != 0
print_result("CPU (N = 0) корректный отказ:", is_cpu_0_ok)

_, out, err = run_process("pi_gpu.exe --method midpoint --n 0 --block 256")
is_gpu_0_ok = "error" in out.lower() or "error" in err.lower() or _ != 0
print_result("GPU (N = 0) корректный отказ:", is_gpu_0_ok)

# Тест 2: Точность метода прямоугольников
print("\n--- Тест 2: Точность метода прямоугольников (midpoint) ---")
for n_val in[10, 100, 1000]:
    sum_py = sum(1.0 / (1.0 + ((i + 0.5) / n_val)**2) for i in range(n_val))
    pi_expected = 4.0 * sum_py / n_val
    
    _, out_cpu, _ = run_process(f"pi_cpu.exe --method midpoint --mode serial --n {n_val}")
    _, out_gpu, _ = run_process(f"pi_gpu.exe --method midpoint --n {n_val} --block 32")
    
    try:
        pi_cpu = float(out_cpu.split(",")[-2])
        cpu_ok = math.isclose(pi_cpu, pi_expected, rel_tol=1e-9)
    except:
        cpu_ok, pi_cpu = False, 0.0
        
    try:
        pi_gpu = float(out_gpu.split(",")[-2])
        gpu_ok = math.isclose(pi_gpu, pi_expected, rel_tol=1e-9)
    except:
        gpu_ok, pi_gpu = False, 0.0

    print_result(f"CPU Midpoint N={n_val}:", cpu_ok, f"(Вычислено: {pi_cpu:.10f}, Ожидалось: {pi_expected:.10f})")
    print_result(f"GPU Midpoint N={n_val}:", gpu_ok, f"(Вычислено: {pi_gpu:.10f}, Ожидалось: {pi_expected:.10f})")

# Тест 3: Воспроизводимость Монте-Карло
print("\n--- Тест 3: Воспроизводимость Monte Carlo ---")
seed_test = 777
_, out_mc1, _ = run_process(f"pi_cpu.exe --method monte_carlo --mode serial --n 1000 --seed {seed_test}")
_, out_mc2, _ = run_process(f"pi_cpu.exe --method monte_carlo --mode serial --n 1000 --seed {seed_test}")
try:
    val1 = float(out_mc1.split(",")[-2])
    val2 = float(out_mc2.split(",")[-2])
    mc_reproducible = (val1 == val2)
except:
    mc_reproducible = False
print_result("CPU MC Воспроизводимость при одном seed:", mc_reproducible)

print("\n================ ПРОВЕРКА ЗАВЕРШЕНА ================\n")

# ======================================================================
# ЭТАП 2: МАССОВЫЕ ИЗМЕРЕНИЯ И СБОР СТАТИСТИКИ (РАЗДЕЛ 5)
# ======================================================================
print("=== ЭТАП 2: ЗАПУСК МАССОВОГО ЭКСПЕРИМЕНТА ===")
print(f"Потоки CPU: {cpu_threads_filtered} | Блоки GPU: {GPU_BLOCK_SIZES}")

random.seed(42)
seeds_for_repeats = [random.randint(1000, 9999) for _ in range(NUM_REPEATS + WARMUP_REPEATS)]

header = ["method", "device", "mode", "n", "threads", "block_size", "seed", "repeat", "time_ms", "kernel_ms", "pi_value", "abs_error", "hardware_id"]
with open(csv_filepath, mode="w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerow(header) # ИСПРАВЛЕНО ТУТ

for n in N_LIST:
    print(f"Выполняются замеры для N = {n}...")
    for method in ["midpoint", "monte_carlo"]:
        
        # 1. CPU Serial
        for r in range(WARMUP_REPEATS + NUM_REPEATS):
            s = seeds_for_repeats[r] if method == "monte_carlo" else 1234
            cmd = f"pi_cpu.exe --method {method} --mode serial --n {n} --seed {s}"
            execute_and_save(cmd, r < WARMUP_REPEATS, r - WARMUP_REPEATS, seeds_for_repeats, r, method)

        # 2. CPU OpenMP
        for threads in cpu_threads_filtered:
            for r in range(WARMUP_REPEATS + NUM_REPEATS):
                s = seeds_for_repeats[r] if method == "monte_carlo" else 1234
                cmd = f"pi_cpu.exe --method {method} --mode omp --n {n} --threads {threads} --seed {s}"
                execute_and_save(cmd, r < WARMUP_REPEATS, r - WARMUP_REPEATS, seeds_for_repeats, r, method)

        # 3. GPU CUDA
        for block_size in GPU_BLOCK_SIZES:
            for r in range(WARMUP_REPEATS + NUM_REPEATS):
                s = seeds_for_repeats[r] if method == "monte_carlo" else 1234
                cmd = f"pi_gpu.exe --method {method} --n {n} --block {block_size} --seed {s}"
                execute_and_save(cmd, r < WARMUP_REPEATS, r - WARMUP_REPEATS, seeds_for_repeats, r, method)

print(f"\n[УСПЕШНО] Все данные сохранены в {csv_filepath}")
