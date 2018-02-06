/*
 * @Name: gpu_info.cu
 * @Description: Prints information about local GPUs.
 *
 * @Author: Giacomo Marciani <gmarciani@acm.org>
 * @Institution: University of Rome Tor Vergata
 */

#include <stdio.h>
#include "../common/error.h"

int main(void) {
  cudaDeviceProp prop; // properties
  int numDevices; // number of devices

  HANDLE_ERROR(cudaGetDeviceCount(&numDevices));

  for (int i = 0; i < numDevices; i++) {
    // get properties for i-th device
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, i));

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

    printf("\n");
  }
}
