import os
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Настраиваем пути
ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "results" / "experiment_results.csv"
PLOT_DIR = ROOT / "results" / "plots"


def generate_plots():
    print("=== Построение графиков ===")
    
    # Проверяем, есть ли вообще файл с данными
    if not CSV_PATH.exists():
        sys.exit(f"Ошибка: Файл данных не найден!\nСначала запустите parallel_ga.exe, чтобы создать {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    max_p = df["P"].max()
    max_d = df["D"].max()

    base_cpu = df[(df["P"] == max_p) & (df["D"] == max_d) &
                  (df["Mode"] == "OpenMP") & (df["Threads"] == 1)].iloc[0]
    t_kernel_base = base_cpu["KernelTimeMs"]
    t_total_base = base_cpu["TotalTimeMs"]

    omp = df[(df["P"] == max_p) & (df["D"] == max_d) &
             (df["Mode"] == "OpenMP")].sort_values("Threads")
    cuda = df[(df["P"] == max_p) & (df["D"] == max_d) &
              (df["Mode"] == "CUDA")].iloc[0]

    threads = omp["Threads"].values
    max_threads = int(threads[-1])

    omp_kernel_speedup = t_kernel_base / omp["KernelTimeMs"].values
    omp_total_speedup = t_total_base / omp["TotalTimeMs"].values
    cuda_kernel_speedup = t_kernel_base / cuda["KernelTimeMs"]
    cuda_total_speedup = t_total_base / cuda["TotalTimeMs"]

    # --- ГРАФИК 1: Ядро ---
    plt.figure(figsize=(10, 6))
    plt.plot(threads, omp_kernel_speedup, "o-", label="OpenMP (ядро)", color="blue", linewidth=2)
    plt.plot(threads, threads, "--", label="Идеальное ускорение", color="gray")
    plt.scatter([max_threads], [cuda_kernel_speedup], color="red", marker="*", s=200, zorder=5,
                label=f"CUDA GPU ядро, T_kernel ({cuda_kernel_speedup:.1f}x)")
    plt.title(f"Ускорение вычислительного ядра Растригина (P={max_p}, D={max_d})")
    plt.xlabel("Количество потоков CPU")
    plt.ylabel("Ускорение (Раз)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "01_kernel_speedup.png", dpi=300)
    plt.close()

    # --- ГРАФИК 2: Полное время ---
    plt.figure(figsize=(10, 6))
    plt.plot(threads, omp_total_speedup, "s-", label="OpenMP (полное время)", color="green", linewidth=2)
    plt.scatter([max_threads], [cuda_total_speedup], color="darkorange", marker="X", s=200, zorder=5,
                label=f"CUDA GPU (T_total, {cuda_total_speedup:.2f}x) — с учётом PCIe")
    plt.axhline(1.0, linestyle=":", color="gray", label="Базовая линия (1x)")
    plt.title(f"Общее ускорение алгоритма ГА (P={max_p}, D={max_d})")
    plt.xlabel("Количество потоков CPU")
    plt.ylabel("Ускорение (Раз)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "02_total_speedup.png", dpi=300)
    plt.close()

    # --- ГРАФИК 3: Разложение CUDA ---
    plt.figure(figsize=(10, 6))
    labels = [
        "CPU OpenMP\n(1 поток)",
        f"CPU OpenMP\n({max_threads} потоков)",
        "CUDA\n(T_kernel)",
        "CUDA\n(T_total)",
    ]
    values = [
        t_kernel_base,
        omp["KernelTimeMs"].iloc[-1],
        cuda["KernelTimeMs"],
        cuda["TotalTimeMs"],
    ]
    colors = ["steelblue", "steelblue", "darkorange", "firebrick"]
    bars = plt.bar(labels, values, color=colors)
    for bar, v in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{v:.1f} ms", ha="center", va="bottom", fontsize=10)
    plt.title(f"Сравнение T_kernel и T_total (P={max_p}, D={max_d})")
    plt.ylabel("Время, мс")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "03_cuda_breakdown.png", dpi=300)
    plt.close()

    print("Готово! Графики сохранены в папку results/plots/:")
    for f in sorted(PLOT_DIR.glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    generate_plots()