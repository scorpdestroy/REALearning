import os
import subprocess
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Конфигурация согласно матрице измерений (раздел 6.1)
N_LIST = [100, 1000, 10000, 100000]
DATASETS = ["MIX", "EASY", "HARD"]
THREADS_LIST = [1, 2, 4, 8]
BLOCK_SIZES = [128, 256]
REPEATS = 7
SEED = 2026

CSV_FILE = "results/raw_data.csv"
os.makedirs("results", exist_ok=True)

# Заголовки по методичке (раздел 7.1)
CSV_HEADERS = [
    "machine_id", "algorithm", "mode", "dataset", "n", 
    "threads", "block", "seed", "repeat", "wall_ms", "kernel_ms", "prime_count", "correct"
]

print("[INFO] Начало вычислительных экспериментов...")

results = []

# Вспомогательная функция для разбора строки вывода программы
def parse_output(output_str, repeat_idx):
    # Программы выводят строку вида: CPU_Local,mode,schedule,dataset,n,threads,block,seed,1,wall_ms,kernel_ms,count,1
    parts = output_str.strip().split(',')
    if len(parts) >= 13:
        parts[8] = str(repeat_idx) # подменяем номер повтора
        return parts
    return None

# --- 1. Прогон CPU версий ---
if os.path.exists("cpu_a1.exe"):
    for n in N_LIST:
        for ds in DATASETS:
            # Serial режим
            for r in range(1, REPEATS + 1):
                cmd = f"cpu_a1.exe --mode serial --count {n} --dataset {ds} --seed {SEED}"
                out = subprocess.check_output(cmd, shell=True).decode('utf-8')
                parsed = parse_output(out, r)
                if parsed: results.append(parsed)
            
            # OpenMP режим с разным числом потоков
            for t in THREADS_LIST:
                for r in range(1, REPEATS + 1):
                    cmd = f"cpu_a1.exe --mode omp --count {n} --threads {t} --dataset {ds} --seed {SEED} --schedule static"
                    out = subprocess.check_output(cmd, shell=True).decode('utf-8')
                    parsed = parse_output(out, r)
                    if parsed: results.append(parsed)

# --- 2. Прогон GPU (CUDA) версий ---
if os.path.exists("gpu_a1.exe"):
    for n in N_LIST:
        for ds in DATASETS:
            for b in BLOCK_SIZES:
                for r in range(1, REPEATS + 1):
                    cmd = f"gpu_a1.exe --count {n} --block {b} --dataset {ds} --seed {SEED}"
                    out = subprocess.check_output(cmd, shell=True).decode('utf-8')
                    parsed = parse_output(out, r)
                    if parsed: results.append(parsed)

# Сохранение в CSV
with open(CSV_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(CSV_HEADERS)
    writer.writerows(results)

print(f"[SUCCESS] Все данные сохранены в {CSV_FILE}")

# --- 3. Построение графиков (Раздел 6.4) ---
df = pd.read_csv(CSV_FILE)
# Фильтруем данные: отбрасываем первый запуск (прогрев), берем медиану по остальным
df_filtered = df[df['repeat'] > 1]
df_med = df_filtered.groupby(['algorithm', 'mode', 'dataset', 'n', 'threads', 'block'], dropna=False).median(numeric_only=True).reset_index()


plt.figure(figsize=(12, 10))

# График 1: Время обработки от N (набор MIX)
plt.subplot(2, 2, 1)
mix_data = df_med[df_med['dataset'] == 'MIX']
# Serial
s_data = mix_data[mix_data['algorithm'] == 'serial']
plt.plot(s_data['n'], s_data['wall_ms'], label='CPU Serial', marker='o')
# OMP 4 threads
omp_data = mix_data[(mix_data['algorithm'] == 'omp') & (mix_data['threads'] == 4)]
plt.plot(omp_data['n'], omp_data['wall_ms'], label='CPU OMP (4 th)', marker='s')
# CUDA
cuda_data = mix_data[(mix_data['algorithm'] == 'cuda') & (mix_data['block'] == 256)]
plt.plot(cuda_data['n'], cuda_data['wall_ms'], label='GPU End-to-End', marker='^')
plt.plot(cuda_data['n'], cuda_data['kernel_ms'], label='GPU Kernel Only', marker='x', linestyle='--')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Размер массива N')
plt.ylabel('Время (мс)')
plt.title('Время обработки от N (MIX)')
plt.legend()
plt.grid(True)

# График 2: Ускорение OpenMP от числа потоков (N=100000, MIX)
plt.subplot(2, 2, 2)
n_max = 100000
t_serial = df_med[(df_med['algorithm'] == 'serial') & (df_med['n'] == n_max) & (df_med['dataset'] == 'MIX')]['wall_ms'].values[0]
omp_threads_data = df_med[(df_med['algorithm'] == 'omp') & (df_med['n'] == n_max) & (df_med['dataset'] == 'MIX')]
omp_threads_data = omp_threads_data.sort_values('threads')
speedup = t_serial / omp_threads_data['wall_ms']
plt.plot(omp_threads_data['threads'], speedup, marker='o', color='g', label='Эксперимент')
plt.plot(omp_threads_data['threads'], omp_threads_data['threads'], linestyle='--', color='gray', label='Идеальное')
plt.xlabel('Число потоков CPU')
plt.ylabel('Ускорение S')
plt.title(f'Ускорение OpenMP от числа потоков (N={n_max})')
plt.legend()
plt.grid(True)

# График 3: Пропускная способность от N для разных профилей нагрузки (Для CUDA End-to-End)
plt.subplot(2, 2, 3)
cuda_all = df_med[(df_med['algorithm'] == 'cuda') & (df_med['block'] == 256)]
for ds in DATASETS:
    ds_data = cuda_all[cuda_all['dataset'] == ds].sort_values('n')
    throughput = ds_data['n'] / (ds_data['wall_ms'] / 1000.0) # проверок в секунду
    plt.plot(ds_data['n'], throughput, label=f'GPU E2E ({ds})', marker='o')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Размер массива N')
plt.ylabel('Пропускная способность (проверок/с)')
plt.title('Пропускная способность под разной нагрузкой')
plt.legend()
plt.grid(True)

# График 4: Влияние размера блока CUDA на время ядра (N=100000, HARD)
plt.subplot(2, 2, 4)
block_data = df_med[(df_med['algorithm'] == 'cuda') & (df_med['n'] == n_max) & (df_med['dataset'] == 'HARD')].sort_values('block')

# Проверяем, что данные вообще есть
if not block_data.empty:
    # 1. Строим столбики
    bars = plt.bar(block_data['block'].astype(str), block_data['kernel_ms'], color='orange', width=0.4)

    plt.xlabel('Размер CUDA блока')
    plt.ylabel('Время ядра (мс)')
    plt.title(f'Влияние блока CUDA на Kernel Time (N={n_max}, HARD)')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    # 2. БЕЗОПАСНЫЙ РАСЧЕТ ОСИ Y
    y_min = block_data['kernel_ms'].min()
    y_max = block_data['kernel_ms'].max()
    diff = y_max - y_min

    # Если разница слишком мала или равна 0, делаем фиксированный отступ в 10%
    if diff < 1e-4:
        diff = y_min * 0.1 if y_min > 0 else 1.0

    plt.ylim(max(0, y_min - diff * 0.5), y_max + diff * 0.5)

    # 3. ДОБАВЛЯЕМ ТОЧНЫЕ ЗНАЧЕНИЯ НАД СТОЛБИКАМИ
    for bar in bars:
        yval = bar.get_height()
        if not np.isnan(yval):
            plt.text(
                bar.get_x() + bar.get_width()/2.0, 
                yval + diff * 0.05, 
                f"{yval:.2f} мс", 
                ha='center', 
                va='bottom', 
                fontweight='bold'
            )
else:
    plt.text(0.5, 0.5, 'Нет данных для GPU (HARD)', ha='center', va='center')
    plt.title('Влияние блока CUDA на Kernel Time')
