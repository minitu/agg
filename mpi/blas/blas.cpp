#include <mpi.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>
#include <chrono>
#include "common.h"
#include "params.h"
#include "comp.h"

extern void runCuda(Comp*, Params*, cudaStream_t, cublasHandle_t, int);

int main(int argc, char** argv)
{
  // Start global timer (from before MPI_Init to after MPI_Finalize)
  auto global_start = std::chrono::system_clock::now();

  // Initialize MPI
  MPI_Init(&argc, &argv);
  int rank, tag = 99, n_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

#if DEBUG
  // Print MPI ranks
  if (rank == 0)
    std::cout << "Total number of MPI ranks = " << n_ranks << "\n";
  std::cout << "[MPI " << rank << "] created" << std::endl;
#endif

  // Process parameters and create computation object
  Params* params = new Params(argc, argv, rank);
  Comp* comp = new Comp(params, rank, n_ranks);

  // Create CUDA stream
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // Prepare cuBLAS (if needed)
  cublasHandle_t handle;
  if (params->cublas) {
    cublasCreate(&handle);
    cublasSetStream(handle, stream);
  }

  // Randomize input values
  comp->randomInit();

  // Start timer (only for communication and computation)
  auto com_start = std::chrono::system_clock::now();

  // Perform communication and computation
  if (comp->type == CompType::DOT_GLOBAL) {
    if (!(comp->agg)) {
      // Offload each rank independently
      runCuda(comp, params, stream, handle, rank);

      // Reduce resulting values to get final value
      // FIXME MPI_FLOAT
      Real reduced;
      MPI_Reduce(comp->h_C, &reduced, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
      if (rank == 0)
        std::cout << "Result: " << std::fixed << std::setprecision(2)
          << reduced << std::endl;
    }
    else {
      // Gather input data
      // FIXME MPI_FLOAT
      MPI_Gather(comp->h_A, comp->N, MPI_FLOAT, comp->h_GA, comp->N, MPI_FLOAT,
          0, MPI_COMM_WORLD);
      MPI_Gather(comp->h_B, comp->N, MPI_FLOAT, comp->h_GB, comp->N, MPI_FLOAT,
          0, MPI_COMM_WORLD);

      // Offload as one big kernel
      if (rank == 0) {
        runCuda(comp, params, stream, handle, rank);

        std::cout << "Result: " << std::fixed << std::setprecision(2)
          << *(comp->h_GC) << "\n" << std::endl;
      }
    }
  }
  else if (comp->type == CompType::DOT_LOCAL) {
    // TODO
  }

  // End timer
  auto com_end = std::chrono::system_clock::now();
  std::chrono::duration<double> com_duration = com_end - com_start;

  if (rank == 0) {
    std::cout << "[Timer] Com: " << com_duration.count() << "s" << std::endl;
  }

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

  // End global timer
  auto global_end = std::chrono::system_clock::now();
  std::chrono::duration<double> global_duration = global_end - global_start;

  if (rank == 0) {
    std::cout << "[Timer] Global: " << global_duration.count() << "s\n"
      << std::endl;
  }

  return 0;
}
