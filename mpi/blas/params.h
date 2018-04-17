#ifndef BLAS_PARAMS_H_
#define BLAS_PARAMS_H_

#include "common.h"

struct Params {
  CompType type; // which computation?
  int n_per_dim; // number of elements per dimension
  bool cublas; // use cuBLAS?

  Params(int argc, char** argv);
};

#endif // BLAS_PARAMS_H_
