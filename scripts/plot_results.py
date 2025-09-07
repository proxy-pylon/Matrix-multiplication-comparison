#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt

cpp = pd.read_csv("results/cpp_bench.csv")
py  = pd.read_csv("results/python_bench.csv")
df = pd.concat([cpp, py], ignore_index=True)

# Plot 1: time vs n
plt.figure()
for m in df['method'].unique():
    dd = df[df['method'] == m].sort_values('n')
    plt.plot(dd['n'], dd['seconds_per_multiply'].astype(float), label=m)
plt.xlabel("n")
plt.ylabel("seconds per multiply")
plt.title("Time vs n")
plt.legend()
plt.savefig("plots/time_vs_n.png", dpi=160)

# Plot 2: (time / n^3) vs n to visually test O(n^3)
plt.figure()
for m in df['method'].unique():
    dd = df[df['method'] == m].sort_values('n').copy()
    dd['tn3'] = dd['seconds_per_multiply'].astype(float) / (dd['n']**3)
    plt.plot(dd['n'], dd['tn3'], label=m)
plt.xlabel("n")
plt.ylabel("time / n^3 (s / n^3)")
plt.title("Scaling check: time / n^3 vs n")
plt.legend()
plt.savefig("plots/time_over_n3.png", dpi=160)

# Plot 3: GFLOPS vs n
plt.figure()
for m in df['method'].unique():
    dd = df[df['method'] == m].sort_values('n')
    plt.plot(dd['n'], dd['gflops'].astype(float), label=m)
plt.xlabel("n")
plt.ylabel("GFLOPS")
plt.title("Achieved GFLOPS vs n")
plt.legend()
plt.savefig("plots/gflops_vs_n.png", dpi=160)

