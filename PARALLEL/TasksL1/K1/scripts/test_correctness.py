#!/usr/bin/env python3
"""5.1 — Автоматические тесты корректности.

Запуск из K1/:
    python scripts/test_correctness.py
    python scripts/test_correctness.py --no-cuda
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CPU_EXE = ROOT / "sort_cpu.exe"
GPU_EXE = ROOT / "sort_cuda.exe"

DISTRIBS = ["uniform", "sorted", "reverse", "fewunique", "nearlysorted"]

class Checker:
    def __init__(self):
        self.total = 0
        self.fail = 0

    def run(self, desc, argv, expect_exit=0):
        self.total += 1
        try:
            res = subprocess.run(argv, capture_output=True, text=True, timeout=300)
        except Exception as e:
            print(f"  FAIL {desc}: {e}")
            self.fail += 1
            return

        ok = (res.returncode == expect_exit)

        if ok and expect_exit == 0:
            rows = [ln for ln in res.stdout.splitlines() if ln.startswith("local,")]
            if not rows:
                ok = False
            else:
                for r in rows:
                    if r.split(",")[-1] != "1":
                        ok = False
                        break

        if ok:
            print(f"  OK   {desc}")
        else:
            print(f"  FAIL {desc}  (exit={res.returncode})")
            if res.stderr:
                print("       stderr:", res.stderr.strip()[:200])
            self.fail += 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cuda", action="store_true")
    args = ap.parse_args()

    has_gpu = GPU_EXE.exists() and not args.no_cuda
    c = Checker()

    print("=== 5.1  Тесты корректности (раздел 7) ===\n")

    for d in DISTRIBS:
        for n in (2, 4, 8, 16):
            c.run(f"cpu serial N={n} {d}",
                  [str(CPU_EXE), "--mode", "serial", "--n", str(n), "--input", d, "--seed", "2026"])
            c.run(f"cpu omp    N={n} {d}",
                  [str(CPU_EXE), "--mode", "omp", "--threads", "4",
                   "--n", str(n), "--input", d, "--seed", "2026"])
            c.run(f"cpu std    N={n} {d}",
                  [str(CPU_EXE), "--mode", "std", "--n", str(n), "--input", d, "--seed", "2026"])
            if has_gpu:
                c.run(f"gpu cuda   N={n} {d}",
                      [str(GPU_EXE), "--mode", "cuda", "--block", "256",
                       "--n", str(n), "--input", d, "--seed", "2026"])

    c.run("cpu serial N=0",
          [str(CPU_EXE), "--mode", "serial", "--n", "0", "--input", "uniform", "--seed", "2026"], 2)
    c.run("cpu serial N=1000 (не 2^m)",
          [str(CPU_EXE), "--mode", "serial", "--n", "1000", "--input", "uniform", "--seed", "2026"], 2)
    if has_gpu:
        c.run("gpu cuda N=0",
              [str(GPU_EXE), "--mode", "cuda", "--block", "256",
               "--n", "0", "--input", "uniform", "--seed", "2026"], 2)
        c.run("gpu cuda N=1000",
              [str(GPU_EXE), "--mode", "cuda", "--block", "256",
               "--n", "1000", "--input", "uniform", "--seed", "2026"], 2)
        c.run("gpu cuda block=100",
              [str(GPU_EXE), "--mode", "cuda", "--block", "100",
               "--n", "1024", "--input", "uniform", "--seed", "2026"], 2)

    c.run("cpu serial N=65536 uniform",
          [str(CPU_EXE), "--mode", "serial", "--n", "65536", "--input", "uniform", "--seed", "2026"])
    c.run("cpu omp    N=65536 uniform",
          [str(CPU_EXE), "--mode", "omp", "--threads", "6",
           "--n", "65536", "--input", "uniform", "--seed", "2026"])
    c.run("cpu std    N=65536 uniform",
          [str(CPU_EXE), "--mode", "std", "--n", "65536", "--input", "uniform", "--seed", "2026"])
    if has_gpu:
        c.run("gpu cuda   N=65536 uniform",
              [str(GPU_EXE), "--mode", "cuda", "--block", "256",
               "--n", "65536", "--input", "uniform", "--seed", "2026"])

    print(f"\n=== Итого: {c.total} проверок, провалов: {c.fail} ===")
    sys.exit(0 if c.fail == 0 else 1)

if __name__ == "__main__":
    main()