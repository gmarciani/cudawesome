#include <stdio.h>
#include <stdlib.h>

#include "gpu_functions.cuh"

__host__ void helloGPU(void) {

  __helloGPU<<< 1, 1 >>>();

  cudaDeviceSynchronize();
}

__global__ void __helloGPU(void) {
  printf("[gpu]> Hello world! (global)\n");
  __helloGPUDevice();
}

__device__ void __helloGPUDevice(void) {
  printf("[gpu]> Hello world! (device)\n");
}
