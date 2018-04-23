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

  Real* h_A;
  Real* h_B;
  Real* h_C;
  Real* d_A;
  Real* d_B;
  Real* d_C;

  Comp(Params* p) {
    type = p->type;
    N = p->n_per_dim;

    switch (type) {
      case CompType::DOT:
        mem_size = N * sizeof(Real);

        // allocate input vectors
        cudaMallocHost(&h_A, mem_size);
        cudaMallocHost(&h_B, mem_size);
        cudaMalloc(&d_A, mem_size);
        cudaMalloc(&d_B, mem_size);

        // allocate output values
        cudaMallocHost(&h_C, sizeof(Real));
        cudaMalloc(&d_C, sizeof(Real));

        // initialize input
        randomize(h_A, N);
        randomize(h_B, N);

        // initialize reduction value
        cudaMemset(d_C, 0, sizeof(Real));
        break;
      case CompType::GEMM:
        mem_size = N * N * sizeof(Real);

        // allocate matrices
        cudaMallocHost(&h_A, mem_size);
        cudaMallocHost(&h_B, mem_size);
        cudaMallocHost(&h_C, mem_size);
        cudaMalloc(&d_A, mem_size);
        cudaMalloc(&d_B, mem_size);
        cudaMalloc(&d_C, mem_size);

        // initialize input
        randomize(h_A, N * N);
        randomize(h_B, N * N);

        // initialize output matrix
        cudaMemset(d_C, 0, mem_size);
        break;
      default:
        break;
    }
  }

  void randomize(Real* data, int n) {
    srand(time(NULL));
    for (int i = 0; i < n; i++) {
      data[i] = rand() / (Real)RAND_MAX;
    }
  }

  void printOne(Real* data) {
    switch (type) {
      case CompType::DOT:
        for (int i = 0; i < N; i++) {
          std::cout << std::fixed << std::setprecision(2) << data[i] << " ";
        }
        std::cout << std::endl;
        break;
      case CompType::GEMM:
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
    std::cout << "[Rank " << rank << "] " << 'A' << std::endl;
      printOne(h_A);
    std::cout << "[Rank " << rank << "] " << 'B' << std::endl;
      printOne(h_B);
    std::cout << "[Rank " << rank << "] " << 'C' << std::endl;
    if (type == CompType::DOT) {
      std::cout << std::fixed << std::setprecision(2) << h_C[0] << std::endl;
    }
    else if (type == CompType::GEMM) {
      printOne(h_C);
    }
  }

  ~Comp() {
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
  }
};

#endif
