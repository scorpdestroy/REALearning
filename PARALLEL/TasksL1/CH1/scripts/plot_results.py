import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Настройка стиля для читаемости графиков в отчете
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
os.makedirs("data", exist_ok=True)

# 1. Загрузка данных
csv_path = os.path.join("data", "measurements.csv")
if not os.path.exists(csv_path):
    print(f"Ошибка: Файл {csv_path} не найден. Сначала запустите скрипт измерений!")
    exit(1)

df = pd.read_csv(csv_path)

# Группировка для графиков времени и ускорения (считаем медиану по повторам)
group_cols = ['method', 'device', 'mode', 'n', 'threads', 'block_size']
# Заменяем NaN на удобные маркеры для корректной группировки
df_filled = df.fillna({'threads': 0, 'block_size': 0})

# ИСПРАВЛЕНО: Добавлен флаг numeric_only=True, чтобы игнорировать текстовое поле hardware_id
median_df = df_filled.groupby(group_cols).median(numeric_only=True).reset_index()

PI_REF = 3.14159265358979323846

# ======================================================================
# ГРАФИК 1: Зависимость общего времени T_total от N (Логарифмическая шкала)
# ======================================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
methods = ['midpoint', 'monte_carlo']
titles = ['Метод прямоугольников (Midpoint)', 'Метод Монте-Карло (Monte Carlo)']

for ax, method, title in zip(axes, methods, titles):
    m_data = median_df[median_df['method'] == method]
    
    # Serial CPU
    serial = m_data[(m_data['mode'] == 'serial') & (m_data['device'] == 'CPU')]
    ax.plot(serial['n'], serial['time_ms'], 'o--', label='CPU Serial', linewidth=2)
    
    # OpenMP CPU (берем максимальное число потоков для наглядности)
    max_threads = m_data[m_data['mode'] == 'omp']['threads'].max()
    omp = m_data[(m_data['mode'] == 'omp') & (m_data['threads'] == max_threads)]
    ax.plot(omp['n'], omp['time_ms'], 's-', label=f'CPU OpenMP ({int(max_threads)} th)', linewidth=2)
    
    # GPU CUDA (берем блок 256 по умолчанию)
    gpu = m_data[(m_data['mode'] == 'cuda') & (m_data['block_size'] == 256)]
    if not gpu.empty:
        ax.plot(gpu['n'], gpu['time_ms'], '^--', label='GPU CUDA (T_total, block 256)', linewidth=2)
        
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Число испытаний N', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)

axes[0].set_ylabel('Общее время выполнения T_total (мс)', fontsize=12)
plt.suptitle('График 1: Зависимость общего времени выполнения от N', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('data/plot1_time_vs_n.png', dpi=200)
plt.close()

# ======================================================================
# ГРАФИК 2: Ускорение CPU (S_p) в зависимости от числа потоков
# ======================================================================
plt.figure(figsize=(10, 6))
target_n = median_df['n'].max()

for method, label, color in zip(methods, ['Прямоугольники', 'Монте-Карло'], ['blue', 'green']):
    sub_data = median_df[(median_df['method'] == method) & (median_df['n'] == target_n)]
    
    t_serial_row = sub_data[(sub_data['mode'] == 'serial') & (sub_data['device'] == 'CPU')]
    if t_serial_row.empty: continue
    t_serial = t_serial_row['time_ms'].values[0]
    
    omp_data = sub_data[sub_data['mode'] == 'omp'].sort_values('threads')
    if omp_data.empty: continue
    
    threads = omp_data['threads'].values
    speedup = t_serial / omp_data['time_ms'].values
    
    plt.plot(threads, speedup, 'o-', label=f'{label} (N = {target_n:.0e})', color=color, linewidth=2)

if 'threads' in locals() and len(threads) > 0:
    plt.plot([threads[0], threads[-1]], [threads[0], threads[-1]], 'k--', label='Идеальное ускорение')

plt.xlabel('Количество потоков CPU (p)', fontsize=12)
plt.ylabel('Ускорение S_p = T_serial / T_parallel', fontsize=12)
plt.title(f'График 2: Ускорение CPU относительно Serial режима (при макс N)', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.savefig('data/plot2_cpu_speedup.png', dpi=200)
plt.close()

# ======================================================================
# ГРАФИК 3: GPU: сравнение T_kernel и T_total (Накладные расходы)
# ======================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, method, title in zip(axes, methods, titles):
    gpu_data = median_df[(median_df['method'] == method) & (median_df['mode'] == 'cuda') & (median_df['block_size'] == 256)]
    if gpu_data.empty: continue
    
    ax.plot(gpu_data['n'], gpu_data['time_ms'], 'ro-', label='T_total (с копированием)', linewidth=2)
    ax.plot(gpu_data['n'], gpu_data['kernel_ms'], 'g*--', label='T_kernel (чистое ядро)', linewidth=2)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Число испытаний N', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)

axes[0].set_ylabel('Время на GPU (мс)', fontsize=11)
plt.suptitle('График 3: Накладные расходы памяти GPU (T_total vs T_kernel)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('data/plot3_gpu_overhead.png', dpi=200)
plt.close()

# ======================================================================
# ГРАФИК 4: Абсолютная ошибка как функция N для двух методов
# ======================================================================
plt.figure(figsize=(10, 6))

for method, label, color in zip(methods, ['Прямоугольники O(N⁻²)', 'Монте-Карло O(N⁻⁰⁵)'], ['blue', 'orange']):
    err_data = median_df[(median_df['method'] == method) & (median_df['mode'] == 'serial')]
    plt.plot(err_data['n'], err_data['abs_error'], 'o-', label=label, color=color, linewidth=2)

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Число испытаний N', fontsize=12)
plt.ylabel('Абсолютная ошибка |π_calc − π_ref|', fontsize=12)
plt.title('График 4: Сходимость и падение погрешности при росте N', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.savefig('data/plot4_error_vs_n.png', dpi=200)
plt.close()

# ======================================================================
# ГРАФИК 5: Распределение оценок Пи для Монте-Карло по независимым seed
# ======================================================================
plt.figure(figsize=(10, 6))
boxplot_n = 10**5
mc_raw_data = df[(df['method'] == 'monte_carlo') & (df['n'] == boxplot_n)]

if not mc_raw_data.empty:
    mc_raw_data = mc_raw_data.copy()
    mc_raw_data['label'] = mc_raw_data['device'] + " " + mc_raw_data['mode']
    
    sns.boxplot(data=mc_raw_data, x='label', y='pi_value', width=0.4, palette='Set2')
    plt.axhline(y=PI_REF, color='red', linestyle='--', linewidth=1.5, label='Эталон π')
    
    plt.xlabel('Конфигурация вычислителя', fontsize=12)
    plt.ylabel('Вычисленное значение π', fontsize=12)
    plt.title(f'График 5: Статистический разброс Монте-Карло (N = {boxplot_n:.0e}, 10 разных seed)', fontsize=13, fontweight='bold')
    plt.legend()
    plt.savefig('data/plot5_mc_boxplot.png', dpi=200)
else:
    print("Предупреждение: Нет данных для построения Boxplot Монте-Карло.")
plt.close()

print("[УСПЕШНО] Все 5 обязательных графиков сохранены в папку 'data/'!")
