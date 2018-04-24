#ifndef BLAS_COMP_H_
#define BLAS_COMP_H_

#include <ctime>
#include <iomanip>
#include "common.h"
#include "params.h"

struct Comp {
  CompType type;
  int N;
  size_t mem_size;
  bool agg;
  int rank;
  int n_ranks;

  Real* h_A;
  Real* h_B;
  Real* h_C;
  Real* d_A;
  Real* d_B;
  Real* d_C;

  // Aggregated data
  Real* h_GA;
  Real* h_GB;
  Real* h_GC;
  Real* d_GA;
  Real* d_GB;
  Real* d_GC;

  Comp(Params* p, int rank_, int n_ranks_) {
    type = p->type;
    N = p->n_per_dim;
    agg = p->agg;
    rank = rank_;
    n_ranks = n_ranks_;

    switch (type) {
      case CompType::DOT_GLOBAL:
      case CompType::DOT_LOCAL:
        mem_size = N * sizeof(Real);

        cudaMallocHost(&h_A, mem_size);
        cudaMallocHost(&h_B, mem_size);
        cudaMallocHost(&h_C, sizeof(Real));
        if (!agg) {
          cudaMalloc(&d_A, mem_size);
          cudaMalloc(&d_B, mem_size);
          cudaMalloc(&d_C, sizeof(Real));
          cudaMemset(d_C, 0, sizeof(Real));
        }
        else if (rank == 0) {
          cudaMallocHost(&h_GA, mem_size * n_ranks);
          cudaMallocHost(&h_GB, mem_size * n_ranks);
          cudaMalloc(&d_GA, mem_size * n_ranks);
          cudaMalloc(&d_GB, mem_size * n_ranks);
          if (type == CompType::DOT_GLOBAL) {
            cudaMallocHost(&h_GC, sizeof(Real));
            cudaMalloc(&d_GC, sizeof(Real));
            cudaMemset(d_GC, 0, sizeof(Real));
          }
          else if (type == CompType::DOT_LOCAL) {
            cudaMallocHost(&h_GC, sizeof(Real) * n_ranks);
            cudaMalloc(&d_GC, sizeof(Real) * n_ranks);
            cudaMemset(d_GC, 0, sizeof(Real) * n_ranks);
          }
        }

        break;
      case CompType::GEMM_GLOBAL:
      case CompType::GEMM_LOCAL:
        // TODO
        mem_size = N * N * sizeof(Real);

        // Allocate matrices
        cudaMallocHost(&h_A, mem_size);
        cudaMallocHost(&h_B, mem_size);
        cudaMallocHost(&h_C, mem_size);
        cudaMalloc(&d_A, mem_size);
        cudaMalloc(&d_B, mem_size);
        cudaMalloc(&d_C, mem_size);

        // Initialize output matrix
        cudaMemset(d_C, 0, mem_size);
        break;
      default:
        break;
    }
  }

  void randomInit() {
    switch (type) {
      case CompType::DOT_GLOBAL:
      case CompType::DOT_LOCAL:
        randomize(h_A, N);
        randomize(h_B, N);

        break;
      case CompType::GEMM_GLOBAL:
      case CompType::GEMM_LOCAL:
        randomize(h_A, N * N);
        randomize(h_B, N * N);

        break;
      default:
        break;
    }
  }

  void randomize(Real* data, int n) {
    srand(time(NULL));
    for (int i = 0; i < n; i++) {
      data[i] = 1.0f;
      //data[i] = rand() / (Real)RAND_MAX;
    }
  }

  void printOne(Real* data) {
    switch (type) {
      case CompType::DOT_GLOBAL:
      case CompType::DOT_LOCAL:
        for (int i = 0; i < N; i++) {
          std::cout << std::fixed << std::setprecision(2) << data[i] << " ";
        }
        std::cout << std::endl;
        break;
      case CompType::GEMM_GLOBAL:
      case CompType::GEMM_LOCAL:
        for (int i = 0; i < N; i++) {
          for (int j = 0; j < N; j++) {
            std::cout << std::fixed << std::setprecision(2) << data[i * N + j]
              << " ";
          }
          std::cout << std::endl;
        }
        break;
    }
  }

  void print(int rank) {
    std::cout << "[Rank " << rank << "] " << "A" << std::endl;
      printOne(h_A);
    std::cout << "\n[Rank " << rank << "] " << "B" << std::endl;
      printOne(h_B);
    std::cout << "\n[Rank " << rank << "] " << "C" << std::endl;
    if (type == CompType::DOT_GLOBAL || type == CompType::DOT_LOCAL) {
      std::cout << std::fixed << std::setprecision(2) << h_C[0] << std::endl;
    }
    else if (type == CompType::GEMM_GLOBAL || type == CompType::GEMM_LOCAL) {
      printOne(h_C);
    }
  }

  ~Comp() {
    switch (type) {
      case CompType::DOT_GLOBAL:
      case CompType::DOT_LOCAL:
        cudaFreeHost(h_A);
        cudaFreeHost(h_B);
        cudaFreeHost(h_C);

        if (!agg) {
          cudaFree(d_A);
          cudaFree(d_B);
          cudaFree(d_C);
        }
        else if (rank == 0) {
          cudaFreeHost(h_GA);
          cudaFreeHost(h_GB);
          cudaFreeHost(h_GC);
          cudaFree(d_GA);
          cudaFree(d_GB);
          cudaFree(d_GC);
        }

        break;
      case CompType::GEMM_GLOBAL:
      case CompType::GEMM_LOCAL:
        cudaFreeHost(h_A);
        cudaFreeHost(h_B);
        cudaFreeHost(h_C);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        break;
      default:
        break;
    }
  }
};

#endif
