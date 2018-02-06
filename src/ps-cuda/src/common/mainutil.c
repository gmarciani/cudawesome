/*
 * @Name: mainutil.c
 * @Description: Utilities for main.
 *
 * @Author: Giacomo Marciani <gmarciani@acm.org>
 *          Gabriele Santi   <gsanti@acm.org>
 *
 * @Institution: University of Rome Tor Vergata
 */

#include "mainutil.h"
#include "error.h"


/*------------------------------------------------------------------------------
   @description   Print GPUs information
   @return  void
   ---------------------------------------------------------------------------*/
void printGpuInfo(void) {
  // GPU information
  cudaDeviceProp prop; // properties
  int numDevices; // number of devices

  HANDLE_ERROR(cudaGetDeviceCount(&numDevices));

  for (int i = 0; i < numDevices; ++i) {
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));

    printf("   --- General Information for device %d ---\n", i);

    printf("Name: %s\n", prop.name);

    printf("Compute capability: %d.%d\n", prop.major, prop.minor);

    printf("Clock rate: %d\n", prop.clockRate);

    printf("Device copy overlap: ");
    if (prop.deviceOverlap) {
      printf("Enabled\n");
    } else {
      printf("Disabled\n");
    }

    printf("Kernel execution timeout: ");
    if (prop.kernelExecTimeoutEnabled) {
      printf("Enabled\n");
    } else {
      printf("Disabled\n");
    }

    printf("   --- Memory Information for device %d ---\n", i);

    printf("Total global memory: %ld\n", prop.totalGlobalMem);

    printf("Total constant memory: %ld\n", prop.totalConstMem);

    printf("Max memory pitch: %ld\n", prop.memPitch);

    printf("Texture Alignment: %ld\n", prop.textureAlignment);

    printf("   --- MP Information for device %d ---\n", i);

    printf("MP count: %d\n", prop.multiProcessorCount);

    printf("Shared memory per MP:  %ld\n", prop.sharedMemPerBlock);

    printf("Registers per MP: %d\n", prop.regsPerBlock);

    printf("Threads in warp: %d\n", prop.warpSize);

    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);

    printf("Max thread dimensions: (%d, %d, %d)\n",
              prop.maxThreadsDim[0],
              prop.maxThreadsDim[1],
              prop.maxThreadsDim[2]);

    printf("Max grid dimensions: (%d, %d, %d)\n",
              prop.maxGridSize[0],
              prop.maxGridSize[1],
              prop.maxGridSize[2]);
  }
}

/*------------------------------------------------------------------------------
  @description      Verifies the prefix sum result.
  @param   input    Input data.
  @param   output   Output data.
  @param   n        Number of elements.
  @return  True, if the result is correct; False, otherwise.
  ----------------------------------------------------------------------------*/
bool isPrefixSumCorrect(int *input, int *output, int n) {
    for ( int i = 0; i < n-1; i++ ) {
        input[i+1] = input[i] + input[i+1];
    }

    int diff = 0;

    for ( int i = 0 ; i < n; i++ ) {
        #ifdef VERBOSE
        printf("Position [%d]  - Value = %d\n", i, output[i]);
        #endif
        diff += input[i] - output[i];
    }

    return diff == 0;
}

/*------------------------------------------------------------------------------
  @description  Test if the given number is a power of two.
  @param   x    The number to test.
  @return  True, if the given number is a power of two; False, otherwise.
  ----------------------------------------------------------------------------*/
bool isPowerOfTwo(int x) {
  return x && !(x & (x - 1));
}
