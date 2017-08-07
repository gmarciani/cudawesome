/*
 * @Program: hello_world.cu
 * @Description: The classic Hello World.
 *
 * @Author: Giacomo Marciani <gmarciani@acm.org>
 * @Institution: University of Rome Tor Vergata
 */

#include <stdlib.h>
#include <stdio.h>

__device__ void helloGPUDevice(void) {
  printf("[gpu]> Hello world! (device)\n");
}

__global__ void helloGPU(void) {
  printf("[gpu]> Hello world! (global)\n");
  helloGPUDevice();
}

__host__ void helloCPUFromHost(void) {
  printf("[cpu]> Hello world! (host)\n");
}

void helloCPU(void) {
  printf("[cpu]> Hello world! (normal)\n");
  helloCPUFromHost();
}

int main(void) {

  helloGPU<<< 1, 1 >>>();

  helloCPU();

  return 0;
}
