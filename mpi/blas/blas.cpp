#include <mpi.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <unistd.h>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <string>
#include "common.h"
#include "params.h"

extern void runCuda(Params*, cudaStream_t, cublasHandle_t);

void printMatrix(Real* matrix, int N, char which, int rank) {
  std::cout << "[Rank " << rank << "] " << which << std::endl;
  N = sqrt(N);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      std::cout << std::fixed << std::setprecision(2) << matrix[i * N + j] << " ";
    }
    std::cout << std::endl;
  }
}

int main(int argc, char** argv)
{
  // Handle command line parameters
  int c;
  std::string type_string;
  Comp type = Comp::DOT;
  int n_elems = 8;
  bool cublas = false;

  while ((c = getopt(argc, argv, "t:n:c")) != -1) {
    switch (c) {
      case 't':
        type_string = optarg;
        if (type_string.compare("dot") == 0)
          type = Comp::DOT;
        else if (type_string.compare("gemm") == 0)
          type = Comp::GEMM;
        else {
          std::cout << "Invalid computation type!" << std::endl;
          return -1;
        }
        break;
      case 'n':
        n_elems = atoi(optarg);
        break;
      case 'c':
        cublas = true;
        break;
      default:
        std::cout << "Usage:\n" <<
          "\t-t {dot, gemm}: vector dot product or matrix multiplication\n" <<
          "\t-n N: N element vector or N x N matrix\n" <<
          "\t-c: use cuBLAS\n" << std::endl;
    }
  }

  // Create parameter object
  if (type == Comp::GEMM)
    n_elems *= n_elems;
  Params* params = new Params(type, cublas, n_elems);

  // Initialize MPI
  MPI_Init(&argc, &argv);
  int rank, tag = 99;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

#if DEBUG
  // Print MPI ranks
  std::cout << "MPI rank " << rank << " created" << std::endl;
#endif

  // Allocate data and randomize
  params->malloc();
  params->randomize();

  // Create CUDA stream
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // Prepare cuBLAS (if needed)
  cublasHandle_t handle;
  if (params->cublas) {
    cublasCreate(&handle);
    cublasSetStream(handle, stream);
  }

  // Invoke data transfers and kernel
  runCuda(params, stream, handle);

#if DEBUG
  // Validate results
  if (params->type == Comp::DOT) {
    // TODO
  }
  else if (params->type == Comp::GEMM) {
    printMatrix(params->h_data[A_IDX], params->n_elems, 'A', rank);
    printMatrix(params->h_data[B_IDX], params->n_elems, 'B', rank);
    printMatrix(params->h_data[C_IDX], params->n_elems, 'C', rank);
  }
#endif

  // Destroy cuBLAS handle
  if (params->cublas) {
    cublasDestroy(handle);
  }

  // Destroy CUDA stream
  cudaStreamDestroy(stream);

  // Deallocate memory
  for (int i = 0; i < N_DATA; i++) {
    cudaFreeHost(params->h_data[i]);
    cudaFree(params->d_data[i]);
  }

  // Finish MPI
  MPI_Finalize();
  return 0;
}
