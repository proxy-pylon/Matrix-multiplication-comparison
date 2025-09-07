// g++ -O3 -march=native -DNDEBUG src/mm_bench.cpp -o mm_bench -lopenblas
#include <vector>
#include <random>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <cmath>
#include <cstring>
#include <functional>
extern "C" {
  #include <cblas.h>
}

using Clock = std::chrono::high_resolution_clock;

static void matmul_naive(const double* A, const double* B, double* C, int n) {
  // C = A * B, textbook i-j-k with poor locality for B
  for (int i = 0; i < n; ++i) {
    double* Ci = C + i* n;
    const double* Ai = A + i* n;
    for (int j = 0; j < n; ++j) {
      double sum = 0.0;
      for (int k = 0; k < n; ++k) {
        sum += Ai[k] * B[k*n + j];
      }
      Ci[j] = sum;
    }
  }
}

static void transpose(const double* B, double* BT, int n) {
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      BT[j*n + i] = B[i*n + j];
}

static void matmul_with_BT(const double* A, const double* BT, double* C, int n) {
  // C = A * B  but we pass in BT = B^T for contiguous access
  for (int i = 0; i < n; ++i) {
    const double* Ai = A + i*n;
    double* Ci = C + i*n;
    for (int j = 0; j < n; ++j) {
      const double* BTj = BT + j*n;
      double sum = 0.0;
      for (int k = 0; k < n; ++k) sum += Ai[k] * BTj[k];
      Ci[j] = sum;
    }
  }
}

static void matmul_blocked(const double* A, const double* B, double* C, int n, int nb) {
  // C = A * B with square blocking
  std::memset(C, 0, sizeof(double)*n*n);
  for (int ii = 0; ii < n; ii += nb) {
    int i_max = std::min(ii + nb, n);
    for (int kk = 0; kk < n; kk += nb) {
      int k_max = std::min(kk + nb, n);
      for (int jj = 0; jj < n; jj += nb) {
        int j_max = std::min(jj + nb, n);
        for (int i = ii; i < i_max; ++i) {
          const double* Ai = A + i*n;
          double* Ci = C + i*n;
          for (int k = kk; k < k_max; ++k) {
            const double aik = Ai[k];
            const double* Bk = B + k*n;
            for (int j = jj; j < j_max; ++j) {
              Ci[j] += aik * Bk[j];
            }
          }
        }
      }
    }
  }
}

static void matmul_blas(const double* A, const double* B, double* C, int n) {
  // C = A * B using BLAS (row-major)
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              n, n, n, 1.0, A, n, B, n, 0.0, C, n);
}

static double time_once(std::function<void()> fn, int reps) {
  auto t0 = Clock::now();
  for (int r = 0; r < reps; ++r) fn();
  auto t1 = Clock::now();
  std::chrono::duration<double> dt = t1 - t0;
  return dt.count() / reps;
}

static int scaled_reps(int n, int base_reps, int ref = 200) {
  // keep per-measurement a few seconds, roughly ~ 1/n^3
  double s = (double)ref / std::max(1, n);
  int reps = (int)std::round(base_reps * s * s * s);
  if (reps < 1) reps = 1;
  return reps;
}

int main(int argc, char** argv) {
  int m = 2000, step = 100, nb = 128, base_reps = 8;
  bool do1 = true, do2 = true, do3 = true, do4 = true;
  std::string csv = "results/cpp_bench.csv";

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need = [&](const char* k){ return a.rfind(k,0)==0 && i+1<argc; };
    if (need("--m")) m = std::stoi(argv[++i]);
    else if (need("--step")) step = std::stoi(argv[++i]);
    else if (need("--nb")) nb = std::stoi(argv[++i]);
    else if (need("--base_reps")) base_reps = std::stoi(argv[++i]);
    else if (need("--csv")) csv = argv[++i];
    else if (need("--methods")) {
      std::string s = argv[++i];
      do1 = do2 = do3 = do4 = false;
      for (char c: s) {
        if (c=='1') do1 = true;
        if (c=='2') do2 = true;
        if (c=='3') do3 = true;
        if (c=='4') do4 = true;
      }
    }
  }

  std::ofstream out(csv);
  out << "lang,method,n,reps,seconds_per_multiply,gflops\n";
  std::mt19937_64 rng(42);
  std::uniform_real_distribution<double> U(0.0, 1.0);

  for (int n = step; n <= m; n += step) {
    std::vector<double> A(n*n), B(n*n), C(n*n), BT;
    for (auto& x: A) x = U(rng);
    for (auto& x: B) x = U(rng);
    if (do2) { BT.resize(n*n); transpose(B.data(), BT.data(), n); }

    const double flops = 2.0*double(n)*n*n - double(n)*n; // exact-ish
    const int reps = scaled_reps(n, base_reps);

    if (do1) {
      double t = time_once([&]{ matmul_naive(A.data(), B.data(), C.data(), n); }, reps);
      out << "cpp,1," << n << "," << reps << "," << std::setprecision(10) << t
          << "," << (flops/t)/1e9 << "\n";
    }
    if (do2) {
      double t = time_once([&]{ matmul_with_BT(A.data(), BT.data(), C.data(), n); }, reps);
      out << "cpp,2," << n << "," << reps << "," << std::setprecision(10) << t
          << "," << (flops/t)/1e9 << "\n";
    }
    if (do3) {
      double t = time_once([&]{ matmul_blocked(A.data(), B.data(), C.data(), n, nb); }, reps);
      out << "cpp,3(nb=" << nb << ")," << n << "," << reps << "," << std::setprecision(10) << t
          << "," << (flops/t)/1e9 << "\n";
    }
    if (do4) {
      double t = time_once([&]{ matmul_blas(A.data(), B.data(), C.data(), n); }, reps);
      out << "cpp,4(blas)," << n << "," << reps << "," << std::setprecision(10) << t
          << "," << (flops/t)/1e9 << "\n";
    }
    std::cerr << "n=" << n << " done\n";
  }
  std::cerr << "Wrote " << csv << "\n";
}

