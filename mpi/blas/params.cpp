#include "params.h"
#include <iostream>
#include <unistd.h>
#include <string>

Params::Params(int argc, char** argv) {
  // Default settings
  type = CompType::DOT;
  n_per_dim = 8;
  cublas = false;

  // Handle command line parameters
  int c;
  std::string type_string;
  while ((c = getopt(argc, argv, "t:n:c")) != -1) {
    switch (c) {
      case 't':
        type_string = optarg;
        if (type_string.compare("dot") == 0)
          type = CompType::DOT;
        else if (type_string.compare("gemm") == 0)
          type = CompType::GEMM;
        else {
          std::cout << "Invalid computation type!" << std::endl;
        }
        break;
      case 'n':
        n_per_dim = atoi(optarg);
        break;
      case 'c':
        cublas = true;
        break;
      default:
        std::cout << "Usage:\n" <<
          "\t-t {dot, gemm}: vector dot product or matrix multiplication\n" <<
          "\t-n N: number of elements per dimension\n" <<
          "\t-c: use cuBLAS\n" << std::endl;
    }
  }

  // Print configuration
  std::cout << "\n[Config]\n" << "Computation: ";
  switch (type) {
    case CompType::DOT:
      std::cout << "vector dot product (DOT)";
      break;
    case CompType::GEMM:
      std::cout << "matrix multiplication (GEMM)";
      break;
    default:
      break;
  }
  std::cout << "\n" << "Number of elements per dimension: " << n_per_dim;
  std::cout << "\n" << "Using cuBLAS: " << std::boolalpha << cublas << "\n"
    << std::endl;
}
