#include <iostream>
#include "common.h"
#include "params.h"
#include "comp.h"

#define N_PER_THREAD 16
#define BLOCK_SIZE 16
#define A_MAT(x,y,N) A[x * N + y]
#define B_MAT(x,y,N) B[x * N + y]
#define C_MAT(x,y,N) C[x * N + y]

__global__ void dotp(Real* A, Real* B, Real* C, int N) {
  int gi = (BLOCK_SIZE * BLOCK_SIZE) * blockIdx.x + threadIdx.x;
  int first_idx = gi * N_PER_THREAD;
  int last_idx = (gi + 1) * N_PER_THREAD - 1;

  if (first_idx < N) {
    Real sum = (Real)0.0;
    for (int i = first_idx; i <= last_idx && i < N; i++) {
      sum += A[i] * B[i];
    }

    atomicAdd(C, sum);
  }
}

__global__ void matmul(Real* A, Real* B, Real* C, int N) {
  int ti = threadIdx.x;
  int tj = threadIdx.y;

  int ci = BLOCK_SIZE * blockIdx.x + ti;
  int cj = BLOCK_SIZE * blockIdx.y + tj;

  if (ci < N && cj < N) {
    Real C_sub = 0.0f;

    int A_j = tj;
    int B_i = ti;

    __shared__ Real s_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ Real s_B[BLOCK_SIZE][BLOCK_SIZE];

    for (int l = 0; l < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; l++) {
      s_A[ti][tj] = A_MAT(ci, A_j, N);
      s_B[ti][tj] = B_MAT(B_i, cj, N);
      __syncthreads();

      for (int q = 0; q < BLOCK_SIZE; q++) {
        C_sub += s_A[ti][q] * s_B[q][tj];
      }

      A_j += BLOCK_SIZE;
      B_i += BLOCK_SIZE;
      __syncthreads();
    }

    C_MAT(ci, cj, N) = C_sub;
  }
}

void runCuda(Comp* comp, Params* params, cudaStream_t stream, cublasHandle_t handle) {
  // Unpack Comp
  Real* h_A = comp->h_A;
  Real* h_B = comp->h_B;
  Real* h_C = comp->h_C;
  Real* d_A = comp->d_A;
  Real* d_B = comp->d_B;
  Real* d_C = comp->d_C;
  int N = comp->N;
  size_t size = comp->mem_size;

  if (!params->cublas) {
    // Use simple handwritten kernel
    dim3 dim_block;
    dim3 dim_grid;

    switch (params->type) {
      case CompType::DOT:
        dim_block = dim3(BLOCK_SIZE * BLOCK_SIZE);
        dim_grid = dim3(ceil((Real)N / (dim_block.x * N_PER_THREAD)));

        cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, stream);

        dotp<<<dim_grid, dim_block, 0, stream>>>(d_A, d_B, d_C, N);

        cudaMemcpyAsync(h_C, d_C, sizeof(Real), cudaMemcpyDeviceToHost, stream);

        break;
      case CompType::GEMM:
        dim_block = dim3(BLOCK_SIZE, BLOCK_SIZE);
        dim_grid = dim3(ceil((Real)N / dim_block.x), ceil((Real)N / dim_block.y));

        cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, stream);

        matmul<<<dim_grid, dim_block, 0, stream>>>(d_A, d_B, d_C, N);

        cudaMemcpyAsync(h_C, d_C, size, cudaMemcpyDeviceToHost, stream);

        break;
    }
  }
  else {
    // Use the cuBLAS library
    switch (params->type) {
      case CompType::DOT:
        cublasSetVectorAsync(N, sizeof(Real), h_A, 1, d_A, 1, stream);
        cublasSetVectorAsync(N, sizeof(Real), h_B, 1, d_B, 1, stream);

        cublasSdot(handle, N, d_A, 1, d_B, 1, h_C);

        break;
      case CompType::GEMM:
        Real alpha = 1.0f;
        Real beta = 0.0f;

        cublasSetMatrixAsync(N, N, sizeof(Real), h_A, N, d_A, N, stream);
        cublasSetMatrixAsync(N, N, sizeof(Real), h_B, N, d_B, N, stream);

        // need to switch A and B due to how cuBLAS sees arrays in Fortran style
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N,
            d_A, N, &beta, d_C, N);

        cublasGetMatrixAsync(N, N, sizeof(Real), d_C, N, h_C, N, stream);
        break;
    }
  }

  cudaStreamSynchronize(stream);

  std::cout << "CUDA: " << cudaGetErrorString(cudaGetLastError())
    << "\n" << std::endl;
}
