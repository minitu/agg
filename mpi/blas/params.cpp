#include "params.h"
#include <iostream>
#include <unistd.h>
#include <string>

Params::Params(int argc, char** argv) {
  // Default settings
  type = CompType::DOT_GLOBAL;
  n_per_dim = 8;
  agg = false;
  cublas = false;

  // Handle command line parameters
  int c;
  std::string type_string;
  while ((c = getopt(argc, argv, "t:n:ac")) != -1) {
    switch (c) {
      case 't':
        type_string = optarg;
        if (type_string.compare("dot") == 0)
          type = CompType::DOT_GLOBAL;
        else if (type_string.compare("gemm") == 0)
          type = CompType::GEMM_GLOBAL;
        else {
          std::cout << "Invalid computation type!" << std::endl;
        }
        break;
      case 'n':
        n_per_dim = atoi(optarg);
        break;
      case 'a':
        agg = true;
        break;
      case 'c':
        cublas = true;
        break;
      default:
        std::cout << "Usage:\n" <<
          "\t-t {dot, gemm}: vector dot product or matrix multiplication\n" <<
          "\t-n N: number of elements per dimension\n" <<
          "\t-a: use kernel aggregation\n" <<
          "\t-c: use cuBLAS\n" << std::endl;
    }
  }

  // Print configuration
  std::cout << "\n[Config]\n" << "Computation: ";
  switch (type) {
    case CompType::DOT_GLOBAL:
      std::cout << "vector dot product (DOT)";
      break;
    case CompType::GEMM_GLOBAL:
      std::cout << "matrix multiplication (GEMM)";
      break;
    default:
      break;
  }
  std::cout << "\n" << "Number of elements per dimension: " << n_per_dim;
  std::cout << "\n" << "Using kernel aggregation: " << std::boolalpha << agg;
  std::cout << "\n" << "Using cuBLAS: " << std::boolalpha << cublas << "\n"
    << std::endl;
}
