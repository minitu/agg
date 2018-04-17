#ifndef BLAS_COMMON_H_
#define BLAS_COMMON_H_

#include "cublas_v2.h"

#define DEBUG 1
#define N_DATA 3
#define A_IDX 0
#define B_IDX 1
#define C_IDX 2

// Computation type
// 1. Vector dot product
// 2. Matrix multiplication
enum class Comp { DOT, GEMM };

// Float or double?
// TODO make this a runtime parameter
typedef float Real;

#endif // BLAS_COMMON_H_
