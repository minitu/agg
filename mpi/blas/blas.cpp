#include <mpi.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>
#include "common.h"
#include "params.h"
#include "comp.h"

extern void runCuda(Comp*, Params*, cudaStream_t, cublasHandle_t);

int main(int argc, char** argv)
{
  // Initialize MPI
  MPI_Init(&argc, &argv);
  int rank, tag = 99;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

#if DEBUG
  // Print MPI ranks
  std::cout << "[MPI] rank " << rank << " created" << std::endl;
#endif

  // Process parameters and create computation object
  Params* params = new Params(argc, argv);
  Comp* comp = new Comp(params);

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
  runCuda(comp, params, stream, handle);

#if DEBUG
  // Validate results
  comp->print(rank);
#endif

  // Destroy cuBLAS handle
  if (params->cublas) {
    cublasDestroy(handle);
  }

  // Destroy CUDA stream
  cudaStreamDestroy(stream);

  // Destroy objects
  delete params;
  delete comp;

  // Finish MPI
  MPI_Finalize();
  return 0;
}
