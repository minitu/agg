#include <mpi.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <ctime>
#include <iostream>
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
      std::cout << matrix[i * N + j] << " ";
    }
    std::cout << std::endl;
  }
}

int main(int argc, char** argv)
{
  // Matrix size (one dimension)
  int N = 8;
  bool use_cublas = false;

  // Handle command line parameters
  if (argc >= 2) {
    N = atoi(argv[1]);
    if (argc >= 3) {
      use_cublas = (atoi(argv[2]) != 0) ? true : false;
    }
  }

  // Initialize MPI
  MPI_Init(&argc, &argv);
  int rank, tag = 99;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Allocate matrices
  float *h_mat[3];
  float *d_mat[3];
  int mem_size = N * N * sizeof(float);
  for (int i = 0; i < 3; i++) {
    cudaMallocHost(&h_mat[i], mem_size);
    cudaMalloc(&d_mat[i], mem_size);
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

  // Randomize entries of matrices
  srand(time(NULL));
  for (int i = 0; i < 2; i++) {
    randomInit(h_mat[i], N * N);
  }

  // Invoke data transfers and kernel
  cudaMatmul(h_mat[0], h_mat[1], h_mat[2], d_mat[0], d_mat[1], d_mat[2], N, mem_size, stream, handle, use_cublas);

#if DEBUG
  // Validate results
  printMatrix(h_mat[0], N, 'A', rank);
  printMatrix(h_mat[1], N, 'B', rank);
  printMatrix(h_mat[2], N, 'C', rank);
#endif

  // Destroy cuBLAS handle
  if (use_cublas) {
    cublasDestroy(handle);
  }

  // Destroy CUDA stream
  cudaStreamDestroy(stream);

  // Deallocate matrices
  for (int i = 0; i < 3; i++) {
    cudaFreeHost(h_mat[i]);
    cudaFree(d_mat[i]);
  }

  // Finish MPI
  MPI_Finalize();
  return 0;
}
