#ifndef BLAS_PARAMS_H_
#define BLAS_PARAMS_H_

#include "common.h"

struct Params {
  CompType type; // which computation type
  Integer n_per_dim; // number of elements per dimension
  bool agg; // use kernel aggregation?
  bool cublas; // use cuBLAS?

  Params(int argc, char** argv, int rank);
};

#endif // BLAS_PARAMS_H_
