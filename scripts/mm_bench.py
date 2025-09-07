#!/usr/bin/env python3
import os, argparse, time, math, csv
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # must be set before importing numpy

import numpy as np

def scaled_reps(n, base_reps, ref=100):
    s = ref / max(1, n)
    r = int(round(base_reps * s * s * s))
    return max(1, r)

def method5_naive(A, B):
    n = A.shape[0]
    C = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += A[i, k] * B[k, j]
            C[i, j] = s
    return C

def method6_numpy(A, B):
    return A @ B

def time_avg(fn, reps):
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    t1 = time.perf_counter()
    return (t1 - t0) / reps

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m5", type=int, default=300, help="max n for method 5")
    ap.add_argument("--step", type=int, default=20)
    ap.add_argument("--base_reps5", type=int, default=50)
    ap.add_argument("--base_reps6", type=int, default=8)
    ap.add_argument("--csv", type=str, default="results/python_bench.csv")
    args = ap.parse_args()

    rng = np.random.default_rng(42)
    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lang","method","n","reps","seconds_per_multiply","gflops"])

        # Method 5: pure Python loops
        for n in range(args.step, args.m5 + 1, args.step):
            A = rng.random((n, n), dtype=np.float64)
            B = rng.random((n, n), dtype=np.float64)
            reps = scaled_reps(n, args.base_reps5, ref=80)
            t = time_avg(lambda: method5_naive(A, B), reps)
            flops = 2.0*n*n*n - n*n
            w.writerow(["python","5",n,reps,f"{t:.10f}", f"{(flops/t)/1e9:.6f}"])
            print(f"[5] n={n} t={t:.4f}s")

        # Method 6: NumPy (BLAS)
        for n in range(100, 2001, 100):
            A = rng.random((n, n), dtype=np.float64)
            B = rng.random((n, n), dtype=np.float64)
            reps = scaled_reps(n, args.base_reps6, ref=200)
            t = time_avg(lambda: method6_numpy(A, B), reps)
            flops = 2.0*n*n*n - n*n
            w.writerow(["python","6(numpy)",n,reps,f"{t:.10f}", f"{(flops/t)/1e9:.6f}"])
            print(f"[6] n={n} t={t:.4f}s")

if __name__ == "__main__":
    main()

