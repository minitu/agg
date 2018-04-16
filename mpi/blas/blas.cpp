#include <mpi.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <unistd.h>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <string>
#include "common.h"

extern void cudaMatmul(float*, float*, float*, float*, float*, float*, int,
    int, cudaStream_t, cublasHandle_t, bool);

void randomInit(float* data, int size) {
  for (int i = 0; i < size; i++) {
    data[i] = rand() / (float)RAND_MAX;
  }
}

void printMatrix(float* matrix, int N, char which, int rank) {
  std::cout << "[Rank " << rank << "] " << which << std::endl;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      std::cout << std::fixed << std::setprecision(2) << matrix[i * N + j] << " ";
    }
    std::cout << std::endl;
  }
}

enum class Comp { DOT, GEMM };

Comp comp_type = Comp::DOT;

int main(int argc, char** argv)
{
  // Number of elements
  // Dot product: vector size
  // Matrix product: matrix size (one dimension)
  int N = 8;
  // Use cuBLAS?
  bool use_cublas = false;

  // Handle command line parameters
  int c;
  std::string type;
  while ((c = getopt(argc, argv, "t:n:c")) != -1) {
    switch (c) {
      case 't':
        type = optarg;
        if (type.compare("dot") == 0)
          comp_type = Comp::DOT;
        else if (type.compare("gemm") == 0)
          comp_type = Comp::GEMM;
        else {
          std::cout << "Invalid computation type!" << std::endl;
          return -1;
        }
        break;
      case 'n':
        N = atoi(optarg);
        break;
      case 'c':
        use_cublas = true;
        break;
    }
  }

  // Initialize MPI
  MPI_Init(&argc, &argv);
  int rank, tag = 99;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

#if DEBUG
  // Print MPI ranks
  std::cout << "MPI rank " << rank << " created" << std::endl;
#endif

  // Allocate memory for data
  float *h_data[3];
  float *d_data[3];
  int element_cnt;
  int mem_size;
  if (comp_type == Comp::DOT) {
    element_cnt = N; // vector
  }
  else if (comp_type == Comp::GEMM) {
    element_cnt = N * N; // matrix
  }
  mem_size = element_cnt * sizeof(float);
  for (int i = 0; i < 3; i++) {
    cudaMallocHost(&h_data[i], mem_size);
    cudaMalloc(&d_data[i], mem_size);
  }

  // Create CUDA stream
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // Prepare cuBLAS (if needed)
  cublasHandle_t handle;
  if (use_cublas) {
    cublasCreate(&handle);
    cublasSetStream(handle, stream);
  }

  // Randomize input data
  srand(time(NULL));
  for (int i = 0; i < 2; i++) {
    randomInit(h_data[i], element_cnt);
  }

  // Invoke data transfers and kernel
  if (comp_type == Comp::DOT) {
    // TODO
  }
  else if (comp_type == Comp::GEMM) {
    cudaMatmul(h_data[0], h_data[1], h_data[2], d_data[0], d_data[1], d_data[2],
        N, mem_size, stream, handle, use_cublas);
  }

#if DEBUG
  // Validate results
  if (comp_type == Comp::DOT) {
    // TODO
  }
  else if (comp_type == Comp::GEMM) {
    printMatrix(h_data[0], N, 'A', rank);
    printMatrix(h_data[1], N, 'B', rank);
    printMatrix(h_data[2], N, 'C', rank);
  }
#endif

  // Destroy cuBLAS handle
  if (use_cublas) {
    cublasDestroy(handle);
  }

  // Destroy CUDA stream
  cudaStreamDestroy(stream);

  // Deallocate memory
  for (int i = 0; i < 3; i++) {
    cudaFreeHost(h_data[i]);
    cudaFree(d_data[i]);
  }

  // Finish MPI
  MPI_Finalize();
  return 0;
}
