#ifndef BLAS_COMMON_H_
#define BLAS_COMMON_H_

#include "cublas_v2.h"

#define DEBUG 1

// Computation type
// 1. Vector dot product
// 2. Matrix multiplication
enum class CompType { DOT, GEMM };

// Float or double?
// TODO make this a runtime parameter
typedef float Real;

#endif // BLAS_COMMON_H_
