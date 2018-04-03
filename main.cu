#include "main.h"

// Tests
// - sMemSize               - no effect
// - 0 stream               - no effect
// - synchronize            - no effect
// - > 1 thread, and block  - no effect
// - > 1 thread
// - > 1 block
// - non-default stream

// TODO
// - Implement multiple streams
// - GPU Manager
// - With data, without data

// How to measure time?
// events
// CPU time
// - clock()
// - gettimeofday
// - c++11
// - clock_gettime
// - cutTimer (deprecated?)

__global__ void empty() {
}

Main::Main(CkArgMsg* m) {
  int sMemSize = 0;
  int numCores = 5120; // TODO query this
  int numThreads = 64; // TODO query this
  int numExperiments =  1000; // default
  int numStreams     = 1;//0000; // default
  cudaStream_t streams[numStreams];

  if (m->argc > 1) {
    numExperiments = atoi(m->argv[1]);
  }

  delete m;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float time;

#if 0
  for (int i = 0; i < numStreams; i++) {
    cudaStreamCreate(&streams[i]);
  }
#endif

  // Warm Up
  for (int i = 0; i < 100; i++) {
    //empty<<<numCores / numThreads, numThreads>>>();
    empty<<<1,1>>>();
  }

  cudaDeviceSynchronize();


  for (int i = 0; i < 1; i++) {
    // 1, 10, 100, 1000, 10000  kernel launch(es)
    /*for (int j = 1; j < 100001; j *= 10)*/ {
#if 0
      cudaEventRecord(start);
      cudaEventRecord(start, streams[i]);
#else
      cudaEventRecord(start, 0);
#endif

      // Should we measure each launch and sum or loop over? loop
      // Need to take averages
      for(int experiment = 0; experiment < numExperiments; experiment++) {
#if 0
        empty<<<numCores / numThreads, numThreads, sMemSize, streams[i]>>>();
        empty<<<numCores / numThreads, numThreads, sMemSize>>>();
        empty<<<numCores / numThreads, numThreads>>>();
        empty<<<1, 1, sMemSize, 0>>>();
#else
        empty<<<numCores / numThreads, numThreads, sMemSize, 0>>>();
#endif
      }
#if 0
      cudaEventRecord(stop);
      cudaEventRecord(stop, streams[i]);
#else
      cudaEventRecord(stop, 0);
#endif

      //cudaDeviceSynchronize();
      cudaEventSynchronize(stop);

      cudaEventElapsedTime(&time, start, stop);

      CkPrintf("Total   Launch Latency: %.4f\tms\n", time);
      CkPrintf("Average Launch Latency: %.1f\tus\n", time / numExperiments * 1000);
    }
  }

  CkExit();
}

#include "main.def.h"
