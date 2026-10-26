#!/usr/bin/env python3
"""5.2–5.4 — Полный прогон эксперимента → data/measurements.csv

Запуск из K1/:
    python scripts/run_all.py
    python scripts/run_all.py --skip-big --no-cuda
"""
import argparse
import subprocess
import sys
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent.parent
CPU_EXE = ROOT / "sort_cpu.exe"
GPU_EXE = ROOT / "sort_cuda.exe"
OUT_CSV = ROOT / "data" / "measurements.csv"

HEADER = ("hardware_id,algorithm,device,input_type,n,threads,block_size,seed,"
          "repeat,time_ms,kernel_ms,transfer_ms,correct")

DISTRIBS = ["uniform", "sorted", "reverse", "fewunique", "nearlysorted"]

def run_once(cmd, timeout=600):
    """Возвращает список строк CSV (только с 'local,')."""
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if res.returncode != 0:
        raise RuntimeError(f"run failed ({res.returncode}): {' '.join(map(str, cmd))}\n"
                           f"stderr: {res.stderr.strip()[:300]}")
    return [ln for ln in res.stdout.splitlines() if ln.startswith("local,")]

def series(csv_file, cmd, warmup, repeats, desc=""):
    """Warmup без записи → repeats прогонов пишутся в CSV."""
    print(f"  {desc}")
    for _ in range(warmup):
        subprocess.run(cmd, capture_output=True, text=True)
    rows = run_once(cmd)

    csv_file.write("\n".join(rows) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-big", action="store_true", help="не делать N=2^20")
    ap.add_argument("--no-cuda", action="store_true")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--cores", type=int, default=12, help="для дефолтного числа OMP-потоков")
    ap.add_argument("--out", default=str(OUT_CSV))
    args = ap.parse_args()

    has_gpu = GPU_EXE.exists() and not args.no_cuda
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not CPU_EXE.exists():
        sys.exit(f"не найден {CPU_EXE}, сначала собери sort_cpu.exe")

    sizes = [1024, 16384, 131072] + ([] if args.skip_big else [1048576])

    total_rows = 0
    t0 = time.time()

    with open(out_path, "w", encoding="ascii", newline="\n") as csv_file:
        csv_file.write(HEADER + "\n")

        print("\n[5.2] Основной прогон: 5 распределений × N × 4 алгоритма")
        omp_default = min(args.cores, 8)
        for d in DISTRIBS:
            for n in sizes:
                run_series = lambda *a: series(csv_file, list(a), args.warmup,
                                               args.repeats, desc=f"{a[0]}")
                base = ["--n", str(n), "--input", d, "--seed", str(args.seed),
                        "--repeats", str(args.repeats)]

                series(csv_file,
                       [str(CPU_EXE), "--mode", "serial", *base],
                       args.warmup, args.repeats,
                       f"serial N={n} {d}")
                series(csv_file,
                       [str(CPU_EXE), "--mode", "std", *base],
                       args.warmup, args.repeats,
                       f"std    N={n} {d}")
                series(csv_file,
                       [str(CPU_EXE), "--mode", "omp", "--threads", str(omp_default), *base],
                       args.warmup, args.repeats,
                       f"omp p={omp_default} N={n} {d}")
                if has_gpu:
                    series(csv_file,
                           [str(GPU_EXE), "--mode", "cuda", "--block", "256", *base],
                           args.warmup, args.repeats,
                           f"cuda b=256 N={n} {d}")

        print("\n[5.3] OpenMP: 1..12 потоков на N=2^14 и 2^17 (uniform)")
        for n in (16384, 131072):
            for p in (1, 2, 4, 6, 8, 12):
                series(csv_file,
                       [str(CPU_EXE), "--mode", "omp", "--threads", str(p),
                        "--n", str(n), "--input", "uniform", "--seed", str(args.seed),
                        "--repeats", str(args.repeats)],
                       args.warmup, args.repeats,
                       f"omp p={p} N={n}")

        if has_gpu:
            print("\n[5.3] CUDA: block = 128/256/512 на N=2^14 и 2^17 (uniform)")
            for n in (16384, 131072):
                for b in (128, 256, 512):
                    series(csv_file,
                           [str(GPU_EXE), "--mode", "cuda", "--block", str(b),
                            "--n", str(n), "--input", "uniform", "--seed", str(args.seed),
                            "--repeats", str(args.repeats)],
                           args.warmup, args.repeats,
                           f"cuda b={b} N={n}")

    with open(out_path, "r", encoding="ascii") as f:
        all_lines = f.read().splitlines()
    data_lines = all_lines[1:]
    bad = [ln for ln in data_lines if ln.split(",")[-1] != "1"]
    elapsed = time.time() - t0

    print(f"\n=== Готово за {elapsed/60:.1f} мин ===")
    print(f"CSV: {out_path}")
    print(f"Строк данных: {len(data_lines)}")
    if bad:
        print(f"ПРОВАЛОВ: {len(bad)}", file=sys.stderr)
        for ln in bad[:5]:
            print("  ", ln, file=sys.stderr)
        sys.exit(1)
    print("Все строки correct=1")

if __name__ == "__main__":
    main()