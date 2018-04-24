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
  int rank, n_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

  // Process parameters and create computation object
  Params* params = new Params(argc, argv, rank);
  Comp* comp = new Comp(params, rank, n_ranks);

  // Allocate memory and randomize inputs
  comp->malloc();
  comp->randomInit();

  // Create CUDA stream
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // Prepare cuBLAS (if needed)
  cublasHandle_t handle;
  if (params->cublas) {
    cublasCreate(&handle);
    cublasSetStream(handle, stream);
  }

  // Timers for MPI communication and CUDA computation
  std::chrono::time_point<std::chrono::system_clock> mpi_start;
  std::chrono::time_point<std::chrono::system_clock> mpi_end;
  std::chrono::time_point<std::chrono::system_clock> cuda_start;
  std::chrono::time_point<std::chrono::system_clock> cuda_end;

  // Perform communication and computation
  if (comp->type == CompType::DOT_GLOBAL) {
    if (!(comp->agg)) {
      // Offload each rank independently
      cuda_start = std::chrono::system_clock::now();
      runCuda(comp, params, stream, handle, rank);
      cuda_end = std::chrono::system_clock::now();

      // Reduce resulting values to get final value
      // FIXME MPI_FLOAT
      Real reduced;
      mpi_start = std::chrono::system_clock::now();
      MPI_Reduce(comp->h_C, &reduced, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
      mpi_end = std::chrono::system_clock::now();

      if (rank == 0)
        std::cout << "Result: " << std::fixed << std::setprecision(2)
          << reduced << "\n" << std::endl;
    }
    else {
      // Gather input data
      // FIXME MPI_FLOAT
      mpi_start = std::chrono::system_clock::now();
      MPI_Gather(comp->h_A, comp->N, MPI_FLOAT, comp->h_GA, comp->N, MPI_FLOAT,
          0, MPI_COMM_WORLD);
      MPI_Gather(comp->h_B, comp->N, MPI_FLOAT, comp->h_GB, comp->N, MPI_FLOAT,
          0, MPI_COMM_WORLD);
      mpi_end = std::chrono::system_clock::now();

      // Offload as one big kernel
      cuda_start = std::chrono::system_clock::now();
      if (rank == 0)
        runCuda(comp, params, stream, handle, rank);
      cuda_end = std::chrono::system_clock::now();

      if (rank == 0)
        std::cout << "Result: " << std::fixed << std::setprecision(2)
          << *(comp->h_GC) << "\n" << std::endl;
    }
  }
  else if (comp->type == CompType::DOT_LOCAL) {
    // TODO
  }

  // Print timings
  std::chrono::duration<double> mpi_duration = mpi_end - mpi_start;
  std::chrono::duration<double> cuda_duration = cuda_end - cuda_start;
  if (rank == 0) {
    std::cout << "[Timer] MPI: " << mpi_duration.count() << "s CUDA: "
      << cuda_duration.count() << "s" << std::endl;
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

  // Deallocate memory
  comp->free();

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
