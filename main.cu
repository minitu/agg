#include "main.h"

__global__ void empty() {
}

Main::Main(CkArgMsg* m) {
  delete m;

  // GPU Manager

  // With data, without data

  // How to measure time?

  // 1, 10, 100, 1000, 10000  kernel launch(es)
  for (int i = 0; i < 2; i++) {
    for (int j = 1; j < 100001; j *= 10) {
      empty<<<1,1>>>();
    }
  }

  cudaDeviceSynchronize();

  CkExit();
}

#include "main.def.h"
