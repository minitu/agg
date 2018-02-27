#include "main.h"

Main::Main(CkArgMsg* m) {
  delete m;
  CkPrintf("Hello, World!\n");
}

#include "main.def.h"
