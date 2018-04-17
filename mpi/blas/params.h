#ifndef BLAS_PARAMS_H_
#define BLAS_PARAMS_H_

#include "common.h"

// Contains parameters
struct Params {
  Comp type;
  bool cublas; // use cuBLAS?
  Real* h_data[N_DATA]; // data on host
  Real* d_data[N_DATA]; // data on device
  int n_elems;
  size_t mem_size;

  Params() {}

  Params(Comp type_, bool cublas_, int n_elems_) : type(type_), cublas(cublas_),
    n_elems(n_elems_), mem_size(n_elems_ * sizeof(Real)) {}

  void malloc() {
    switch (type) {
      case Comp::DOT:
        // Vectors for input
        for (int i = 0; i < N_DATA-1; i++) {
          cudaMallocHost(&h_data[i], mem_size);
          cudaMalloc(&d_data[i], mem_size);
        }

        // Single element for output
        cudaMallocHost(&h_data[N_DATA-1], sizeof(Real));
        cudaMalloc(&d_data[N_DATA-1], sizeof(Real));

        break;
      case Comp::GEMM:
        // Input matrices and output matrix
        for (int i = 0; i < N_DATA; i++) {
          cudaMallocHost(&h_data[i], mem_size);
          cudaMalloc(&d_data[i], mem_size);
        }

        break;
      default:
        std::cout << "Wrong type, malloc failed" << std::endl;
        break;
    }
  }

  void randomize() {
    // Randomize input data
    srand(time(NULL));
    for (int i = 0; i < N_DATA-1; i++) {
      for (int j = 0; j < n_elems; j++) {
        h_data[i][j] = rand() / (Real)RAND_MAX;
      }
    }
  }
};

#endif // BLAS_PARAMS_H_
