# Build and Run Guide

## Prerequisites
- Ubuntu 24.04 (WSL works fine)
- GCC with OpenBLAS (`sudo apt install build-essential libopenblas-dev`)
- Python 3.12 with `numpy`, `pandas`, `matplotlib`

## Build C++ code
make

## Run C++ benchmarks
source env_single_thread.sh
./mm_bench --m 2000 --step 100 --nb 128 --base_reps 8 --csv results/cpp.csv

## Run python benchmarks
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib
python3 scripts/mm_bench.py --m5 300 --step 20 --base_reps5 60 --base_reps6 8 --csv results/python.csv

## Generate plots
python3 scripts/plot_results.py


Plots will appear in the plots/ folder.