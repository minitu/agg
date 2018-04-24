#ifndef BLAS_COMMON_H_
#define BLAS_COMMON_H_

#include "cublas_v2.h"
#include <cstdint>

#define DEBUG 0

// Computation type
// 1. Vector dot product
// 2. Matrix multiplication
enum class CompType { DOT_LOCAL, DOT_GLOBAL, GEMM_LOCAL, GEMM_GLOBAL };

// Float or double?
// TODO make this a runtime parameter
typedef float Real;

// 32 or 64 bit integers?
typedef int_fast64_t Integer;

#endif // BLAS_COMMON_H_
