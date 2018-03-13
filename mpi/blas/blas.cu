#include <iostream>
#include "common.h"

#define BLOCK_SIZE 16
#define A_MAT(x,y,N) A[x * N + y]
#define B_MAT(x,y,N) B[x * N + y]
#define C_MAT(x,y,N) C[x * N + y]

__global__ void matmul(float* A, float* B, float* C, int N) {
  int ti = threadIdx.x;
  int tj = threadIdx.y;

  int ci = BLOCK_SIZE * blockIdx.x + ti;
  int cj = BLOCK_SIZE * blockIdx.y + tj;

  if (ci < N && cj < N) {
    float C_sub = 0.0f;

    int A_j = tj;
    int B_i = ti;

    __shared__ float s_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_B[BLOCK_SIZE][BLOCK_SIZE];

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

void cudaMatmul(float* h_A, float* h_B, float* h_C, float* d_A, float* d_B,
    float* d_C, int N, int mem_size, cudaStream_t stream, cublasHandle_t handle,
    bool use_cublas) {
  if (!use_cublas) { // use simple matmul kernel
    dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dim_grid(ceil((float)N / dim_block.x), ceil((float)N / dim_block.y));

    cudaMemcpyAsync(d_A, h_A, mem_size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_B, h_B, mem_size, cudaMemcpyHostToDevice, stream);

    matmul<<<dim_grid, dim_block, 0, stream>>>(d_A, d_B, d_C, N);

    cudaMemcpyAsync(h_C, d_C, mem_size, cudaMemcpyDeviceToHost, stream);
  }
  else { // use cuBLAS
    float alpha = 1.0f;
    float beta = 0.0f;

    cublasSetMatrixAsync(N, N, sizeof(float), h_A, N, d_A, N, stream);
    cublasSetMatrixAsync(N, N, sizeof(float), h_B, N, d_B, N, stream);

    // need to switch A and B due to how cuBLAS sees arrays in Fortran style
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A,
        N, &beta, d_C, N);

    cublasGetMatrixAsync(N, N, sizeof(float), d_C, N, h_C, N, stream);
  }

  cudaStreamSynchronize(stream);

  std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
}
