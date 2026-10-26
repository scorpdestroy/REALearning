#!/usr/bin/env python3
"""6.1–6.2 — Графики и таблицы результатов.

Запуск из K1/:
    python scripts/plots.py
    python scripts/plots.py --run
    python scripts/plots.py --run --test
    python scripts/plots.py --run --force
    python scripts/plots.py --run --skip-big --no-cuda
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["threads"]    = df["threads"].fillna(0).astype(int)
    df["block_size"] = df["block_size"].fillna(0).astype(int)
    df["kernel_ms"]  = pd.to_numeric(df["kernel_ms"], errors="coerce")
    df["transfer_ms"]= pd.to_numeric(df["transfer_ms"], errors="coerce")
    df["time_ms"]    = pd.to_numeric(df["time_ms"], errors="coerce")
    df = df[df["correct"] == 1].copy()
    return df

def plot_time_vs_n(df: pd.DataFrame, out: Path):
    sub = df[(df["input_type"] == "uniform")].copy()
    omp_p = sub.loc[sub["algorithm"] == "bitonic_omp", "threads"]
    if len(omp_p) == 0:
        print("  [1] нет OMP-данных, пропуск"); return
    omp_default = int(omp_p.mode().iloc[0])

    series = {
        "bitonic_serial": sub[sub["algorithm"] == "bitonic_serial"],
        f"bitonic_omp (p={omp_default})":
            sub[(sub["algorithm"] == "bitonic_omp") & (sub["threads"] == omp_default)],
        "std_sort":       sub[sub["algorithm"] == "std_sort"],
        "bitonic_cuda (b=256, T_total)":
            sub[(sub["algorithm"] == "bitonic_cuda") & (sub["block_size"] == 256)],
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, s in series.items():
        if s.empty: continue
        g = s.groupby("n")["time_ms"].median().sort_index()
        ax.plot(g.index, g.values, marker="o", label=label)

    gpu = sub[(sub["algorithm"] == "bitonic_cuda") & (sub["block_size"] == 256)]
    if not gpu.empty:
        gk = gpu.groupby("n")["kernel_ms"].median().sort_index()
        ax.plot(gk.index, gk.values, linestyle="--", marker="x",
                label="bitonic_cuda (b=256, T_kernel)")

    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("N (log2)"); ax.set_ylabel("Время, мс (log)")
    ax.set_title("1. Время сортировки от N — uniform")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "01_time_vs_n.png", dpi=130); plt.close(fig)

def plot_cpu_speedup(df: pd.DataFrame, out: Path):
    omp = df[df["algorithm"] == "bitonic_omp"].copy()
    ser = df[df["algorithm"] == "bitonic_serial"].copy()
    if omp.empty or ser.empty:
        print("  [2] нет данных OMP/serial, пропуск"); return

    Ns = [n for n in sorted(omp["n"].unique()) if n in (16384, 131072)][:2]
    if not Ns:
        print("  [2] нет N=2^14 / 2^17, пропуск"); return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for n in Ns:
        t_ser = ser[ser["n"] == n]["time_ms"].median()
        if pd.isna(t_ser): continue
        sub = omp[omp["n"] == n]
        g = (sub.groupby("threads").agg(t=("time_ms","median")).reset_index()
                .sort_values("threads"))
        p = g["threads"].values.astype(int)
        S = t_ser / g["t"].values
        E = S / p
        axes[0].plot(p, S, marker="o", label=f"N=2^{int(np.log2(n))}")
        axes[1].plot(p, E, marker="s", label=f"N=2^{int(np.log2(n))}")

    axes[0].axhline(1.0, ls=":", color="gray")
    axes[0].set_xlabel("Число потоков p")
    axes[0].set_ylabel("S_CPU(p) = T_serial / T_omp(p)")
    axes[0].set_title("2a. Ускорение OpenMP bitonic")
    axes[0].grid(alpha=0.4); axes[0].legend(fontsize=9)

    axes[1].axhline(1.0, ls=":", color="gray")
    axes[1].set_xlabel("Число потоков p")
    axes[1].set_ylabel("E(p) = S_CPU(p) / p")
    axes[1].set_ylim(0, 1.1)
    axes[1].set_title("2b. Эффективность OpenMP bitonic")
    axes[1].grid(alpha=0.4); axes[1].legend(fontsize=9)
    fig.tight_layout(); fig.savefig(out / "02_cpu_speedup.png", dpi=130); plt.close(fig)

def plot_cuda_timings(df: pd.DataFrame, out: Path):
    gpu = df[(df["algorithm"] == "bitonic_cuda") & (df["input_type"] == "uniform")].copy()
    if gpu.empty:
        print("  [3] нет CUDA-данных, пропуск"); return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sub = gpu[gpu["block_size"] == 256]
    if not sub.empty:
        g = sub.groupby("n").agg(kernel=("kernel_ms","median"),
                                 total=("time_ms","median"),
                                 transfer=("transfer_ms","median")).sort_index()
        axes[0].plot(g.index, g["kernel"], marker="o", label="T_kernel")
        axes[0].plot(g.index, g["transfer"], marker="^", label="T_transfer (H2D+D2H)")
        axes[0].plot(g.index, g["total"], marker="s", label="T_total")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("N (log2)"); axes[0].set_ylabel("мс")
    axes[0].set_title("3a. CUDA: T_kernel / T_transfer / T_total (b=256)")
    axes[0].grid(alpha=0.4); axes[0].legend(fontsize=9)

    for n in sorted(gpu["n"].unique()):
        s = gpu[gpu["n"] == n].groupby("block_size")["time_ms"].median().sort_index()
        if len(s) < 2: continue
        axes[1].plot(s.index, s.values, marker="o", label=f"N=2^{int(np.log2(n))}")
    axes[1].set_xlabel("Размер блока CUDA"); axes[1].set_ylabel("T_total, мс")
    axes[1].set_title("3b. Эффект размера блока CUDA")
    axes[1].grid(alpha=0.4); axes[1].legend(fontsize=9)
    fig.tight_layout(); fig.savefig(out / "03_cuda_timings.png", dpi=130); plt.close(fig)

def plot_distribution_effect(df: pd.DataFrame, out: Path):
    target_n = 131072 if 131072 in df["n"].values else int(df["n"].max())
    sub = df[df["n"] == target_n].copy()
    if sub.empty:
        print("  [4] нет данных, пропуск"); return

    omp_p_series = sub.loc[sub["algorithm"] == "bitonic_omp", "threads"]
    omp_p = int(omp_p_series.mode().iloc[0]) if len(omp_p_series) else 6

    algos = {
        "bitonic_serial": sub[sub["algorithm"] == "bitonic_serial"],
        f"bitonic_omp (p={omp_p})":
            sub[(sub["algorithm"] == "bitonic_omp") & (sub["threads"] == omp_p)],
        "std_sort":       sub[sub["algorithm"] == "std_sort"],
        "bitonic_cuda (b=256)":
            sub[(sub["algorithm"] == "bitonic_cuda") & (sub["block_size"] == 256)],
    }

    dists = ["uniform","sorted","reverse","fewunique","nearlysorted"]
    dists = [d for d in dists if d in sub["input_type"].unique()]

    data = []
    for label, s in algos.items():
        if s.empty: continue
        med = s.groupby("input_type")["time_ms"].median().reindex(dists)
        data.append((label, med))
    if not data:
        print("  [4] пусто, пропуск"); return

    labels = [d[0] for d in data]
    x = np.arange(len(dists)); w = 0.8 / len(labels)

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (label, med) in enumerate(data):
        ax.bar(x + i*w - (len(labels)-1)*w/2, med.values, width=w, label=label)
    ax.set_xticks(x); ax.set_xticklabels(dists, rotation=15)
    ax.set_ylabel("Медиана времени, мс"); ax.set_yscale("log")
    ax.set_title(f"4. Влияние распределения (N=2^{int(np.log2(target_n))})")
    ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "04_distribution_effect.png", dpi=130); plt.close(fig)

def plot_throughput(df: pd.DataFrame, out: Path):
    sub = df[df["input_type"] == "uniform"].copy()
    if sub.empty:
        print("  [5] нет uniform-данных, пропуск"); return
    omp_p_series = sub.loc[sub["algorithm"] == "bitonic_omp", "threads"]
    omp_p = int(omp_p_series.mode().iloc[0]) if len(omp_p_series) else 6

    series = {
        "bitonic_serial": sub[sub["algorithm"] == "bitonic_serial"],
        f"bitonic_omp (p={omp_p})":
            sub[(sub["algorithm"] == "bitonic_omp") & (sub["threads"] == omp_p)],
        "std_sort":       sub[sub["algorithm"] == "std_sort"],
        "bitonic_cuda (b=256)":
            sub[(sub["algorithm"] == "bitonic_cuda") & (sub["block_size"] == 256)],
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, s in series.items():
        if s.empty: continue
        g = s.groupby("n")["time_ms"].median().sort_index()
        q = (g.index.astype(float) / (g.values / 1000.0)) / 1e6
        ax.plot(g.index, q, marker="o", label=label)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("N (log2)"); ax.set_ylabel("Q, млн элементов/с (log)")
    ax.set_title("5. Пропускная способность Q = N / T — uniform")
    ax.grid(True, which="both", ls=":", alpha=0.5); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(out / "05_throughput.png", dpi=130); plt.close(fig)

def build_summary_tables(df: pd.DataFrame, out: Path):
    written = []

    ser = df[(df["algorithm"] == "bitonic_serial") & (df["input_type"] == "uniform")]
    omp = df[(df["algorithm"] == "bitonic_omp")    & (df["input_type"] == "uniform")]
    rows = []
    for n in sorted(omp["n"].unique()):
        t_ser = ser[ser["n"] == n]["time_ms"].median()
        if pd.isna(t_ser): continue
        for p in sorted(omp["threads"].unique()):
            t = omp[(omp["n"] == n) & (omp["threads"] == p)]["time_ms"].median()
            if pd.isna(t): continue
            S = t_ser / t
            rows.append(dict(n=int(n), threads=int(p), T_serial_ms=float(t_ser),
                             T_omp_ms=float(t), S_CPU=float(S), E=float(S / p)))
    if rows:
        p_out = out / "summary_cpu_speedup.csv"
        pd.DataFrame(rows).to_csv(p_out, index=False, float_format="%.4f")
        written.append(p_out)

    uni = df[df["input_type"] == "uniform"].copy()
    ser_med = (uni[uni["algorithm"] == "bitonic_serial"]
               .groupby("n")["time_ms"].median())
    std_med = (uni[uni["algorithm"] == "std_sort"]
               .groupby("n")["time_ms"].median())

    omp_p = int(uni.loc[uni["algorithm"] == "bitonic_omp", "threads"].mode().iloc[0]) \
            if (uni["algorithm"] == "bitonic_omp").any() else 6

    rows = []
    def add(label, med, gpu_metrics=False):
        for n, t in med.items():
            if pd.isna(t): continue
            n_int = int(n)
            Q = (n_int / (t / 1000.0)) / 1e6
            row = dict(algorithm=label, n=n_int, T_ms=float(t),
                       S_CPU=np.nan, S_GPU=np.nan, R_std=np.nan, Q_Meps=float(Q))
            if gpu_metrics:
                row["S_GPU"] = float(ser_med.get(n, np.nan) / t) if n in ser_med else np.nan
                row["R_std"] = float(std_med.get(n, np.nan) / t) if n in std_med else np.nan
            else:
                if label == "bitonic_serial":
                    row["S_CPU"] = 1.0
                elif n in ser_med:
                    row["S_CPU"] = float(ser_med[n] / t)
                if n in std_med:
                    row["R_std"] = float(std_med[n] / t)
            rows.append(row)

    add("bitonic_serial", ser_med.to_dict())
    add("std_sort",       std_med.to_dict())
    add(f"bitonic_omp(p={omp_p})",
        (uni[(uni["algorithm"] == "bitonic_omp") & (uni["threads"] == omp_p)]
         .groupby("n")["time_ms"].median()).to_dict())

    if (uni["algorithm"] == "bitonic_cuda").any():
        for b in sorted(uni.loc[uni["algorithm"] == "bitonic_cuda", "block_size"].unique()):
            b = int(b)
            med = (uni[(uni["algorithm"] == "bitonic_cuda") & (uni["block_size"] == b)]
                   .groupby("n")["time_ms"].median())
            add(f"bitonic_cuda(b={b})", med.to_dict(), gpu_metrics=True)

    if rows:
        p_out = out / "summary_speedups.csv"
        pd.DataFrame(rows)[["algorithm","n","T_ms","S_CPU","S_GPU","R_std","Q_Meps"]] \
            .to_csv(p_out, index=False, float_format="%.4f")
        written.append(p_out)

    dists = ["uniform","sorted","reverse","fewunique","nearlysorted"]
    target_n = 131072 if 131072 in uni["n"].values else int(uni["n"].max())
    base = df[df["n"] == target_n].copy()
    rows = []
    masks = [
        ("bitonic_serial", base["algorithm"] == "bitonic_serial"),
        (f"bitonic_omp(p={omp_p})",
         (base["algorithm"] == "bitonic_omp") & (base["threads"] == omp_p)),
        ("std_sort",       base["algorithm"] == "std_sort"),
        ("bitonic_cuda(b=256)",
         (base["algorithm"] == "bitonic_cuda") & (base["block_size"] == 256)),
    ]
    for label, mask in masks:
        s = base[mask]
        if s.empty: continue
        med = s.groupby("input_type")["time_ms"].median().reindex(dists)
        row = dict(algorithm=label, n=int(target_n))
        for d in dists:
            row[d + "_ms"] = float(med.get(d, np.nan))
        rows.append(row)
    if rows:
        p_out = out / "summary_distribution.csv"
        pd.DataFrame(rows).to_csv(p_out, index=False, float_format="%.3f")
        written.append(p_out)

    return written

def _run_script(name: str, extra_args: list[str], cwd: Path) -> int:
    script = cwd / "scripts" / name
    print(f"\n>>> python scripts/{name} {' '.join(extra_args)}")
    res = subprocess.run([sys.executable, str(script), *extra_args])
    return res.returncode

def _ensure_measurements(args, root: Path, csv_path: Path) -> bool:
    """Возвращает True, если после проверки CSV пригоден для чтения."""
    need_run = args.run or args.force
    if csv_path.exists():
        size = csv_path.stat().st_size

        if size < 100:
            print(f"CSV слишком маленький ({size} B) — запущу run_all.py")
            need_run = True
    else:
        print(f"CSV не найден: {csv_path} — запущу run_all.py")
        need_run = True

    if args.test:
        rc = _run_script("test_correctness.py",
                         ["--no-cuda"] if args.no_cuda else [], root)
        if rc != 0:
            print("test_correctness.py провалился — прерываю", file=sys.stderr)
            return False

    if need_run:
        extra = []
        if args.skip_big: extra.append("--skip-big")
        if args.no_cuda:  extra.append("--no-cuda")
        rc = _run_script("run_all.py", extra, root)
        if rc != 0:
            print("run_all.py провалился — прерываю", file=sys.stderr)
            return False

    if not csv_path.exists() or csv_path.stat().st_size < 100:
        print("После прогона CSV так и не создан/пустой", file=sys.stderr)
        return False
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/measurements.csv")
    ap.add_argument("--out", default="data/plots")
    ap.add_argument("--run", action="store_true",
                    help="сначала запустить scripts/run_all.py")
    ap.add_argument("--test", action="store_true",
                    help="сначала запустить scripts/test_correctness.py")
    ap.add_argument("--force", action="store_true",
                    help="перегенерировать measurements.csv, даже если он есть")
    ap.add_argument("--skip-big", action="store_true",
                    help="передать --skip-big в run_all.py (без N=2^20)")
    ap.add_argument("--no-cuda", action="store_true",
                    help="не запускать CUDA-часть")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    csv_path = (root / args.csv).resolve() if not Path(args.csv).is_absolute() else Path(args.csv)
    out_dir  = (root / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)

    if not _ensure_measurements(args, root, csv_path):
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nCSV: {csv_path}")
    try:
        df = load(csv_path)
    except pd.errors.EmptyDataError:
        print(f"CSV пустой: {csv_path}", file=sys.stderr)
        sys.exit(1)
    print(f"Строк после фильтра correct=1: {len(df)}")
    if df.empty:
        print("Нечего строить: после фильтра correct=1 осталось 0 строк.",
              file=sys.stderr)
        sys.exit(1)

    print("\n[6.1] Графики:")
    plot_time_vs_n(df, out_dir);            print("  -> 01_time_vs_n.png")
    plot_cpu_speedup(df, out_dir);          print("  -> 02_cpu_speedup.png")
    plot_cuda_timings(df, out_dir);         print("  -> 03_cuda_timings.png")
    plot_distribution_effect(df, out_dir);  print("  -> 04_distribution_effect.png")
    plot_throughput(df, out_dir);           print("  -> 05_throughput.png")

    print("\n[6.2] Таблицы:")
    for p in build_summary_tables(df, out_dir.parent):
        print(f"  -> {p}")

if __name__ == "__main__":
    main()