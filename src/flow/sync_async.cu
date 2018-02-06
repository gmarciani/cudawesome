/*
 * @Program: sync_async.cu
 * @Description: Shows the common sync/async behaviour.
 *
 * @Author: Giacomo Marciani <gmarciani@acm.org>
 * @Institution: University of Rome Tor Vergata
 */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

__host__ __device__ void waitClockCycles(const int cycles) {
  clock_t start = clock();
  clock_t now;
  for (;;) {
    now = clock();
    clock_t cycles = now > start ? now - start : now + (0xffffffff - start);
    if (cycles >= cycles) {
      break;
    }
  }
}

__device__ void helloGPUDevice(const int val, const int cycles, int *result) {
  waitClockCycles(cycles);
  printf("[gpu]> (%d) Hello world! (device | delay: %d clock cycles | in: %d)\n", val, cycles, *result);
  *result = val;
}

__global__ void helloGPU(const int val, const int cycles, int *result) {
  waitClockCycles(cycles);
  printf("[gpu]> (%d) Hello world! (global | delay: %d clock cycles | in: %d)\n", val, cycles, *result);
  *result = val;
  helloGPUDevice(val, cycles, result);
}

__host__ void helloCPUFromHost(const int val, const int cycles) {
  waitClockCycles(cycles);
  printf("[cpu]> (%d) Hello world! (host | delay: %d clock cycles)\n", val, cycles);
}

void helloCPU(const int val, const int cycles) {
  waitClockCycles(cycles);
  printf("[cpu]> (%d) Hello world! (normal | delay: %d clock cycles)\n", val, cycles);
  helloCPUFromHost(val, cycles);
}

void flow_1(void) {
  printf("### start FLOW 1 ###\n");

  int result = -1;
  int *dev_result = NULL;

  cudaMalloc((void**)&dev_result, sizeof(int));

  printf("[before memcpy #1] result: %d\n", result);

  cudaMemcpy(dev_result, &result, sizeof(int), cudaMemcpyHostToDevice); // memcpy #1

  printf("[after memcpy #1] result: %d\n", result);

  helloGPU<<< 1, 1 >>>(1, 30000, dev_result); // gpu #1

  helloGPU<<< 1, 1 >>>(2, 10000, dev_result); // gpu #2

  printf("[before deviceSynchronize #1] result: %d\n", result);

  cudaDeviceSynchronize(); // deviceSynchronize #1

  printf("[after deviceSynchronize #1] result: %d\n", result);

  printf("[before memcpy #2] result: %d\n", result);

  cudaMemcpy(&result, dev_result, sizeof(int), cudaMemcpyDeviceToHost); // memcpy #2

  printf("[after memcpy #2] result: %d\n", result);

  printf("### end FLOW 1 ###\n\n");
}

void flow_2(void) {
  printf("### start FLOW 2 ###\n");

  int result = -1;
  int *dev_result = NULL;

  cudaMalloc((void**)&dev_result, sizeof(int));

  printf("[before memcpy #1] result: %d\n", result);

  cudaMemcpy(dev_result, &result, sizeof(int), cudaMemcpyHostToDevice); // memcpy #1

  printf("[after memcpy #1] result: %d\n", result);

  helloGPU<<< 1, 1 >>>(1, 30000, dev_result); // gpu #1

  helloGPU<<< 1, 1 >>>(2, 10000, dev_result); // gpu #2

  printf("[before memcpy #2] result: %d\n", result);

  cudaMemcpy(&result, dev_result, sizeof(int), cudaMemcpyDeviceToHost); // memcpy #2

  printf("[after memcpy #2] result: %d\n", result);

  printf("### end FLOW 2 ###\n\n");
}

void flow_3(void) {
  printf("### start FLOW 3 ###\n");

  int result = -1;
  int *dev_result = NULL;

  cudaMalloc((void**)&dev_result, sizeof(int));

  printf("[before memcpyAsync #1] result: %d\n", result);

  cudaMemcpyAsync(dev_result, &result, sizeof(int), cudaMemcpyHostToDevice); // memcpyAsync #1

  printf("[after memcpyAsync #1] result: %d\n", result);

  helloGPU<<< 1, 1 >>>(1, 30000, dev_result); // gpu #1

  helloGPU<<< 1, 1 >>>(2, 10000, dev_result); // gpu #2

  printf("[before memcpyAsync #2] result: %d\n", result);

  cudaMemcpyAsync(&result, dev_result, sizeof(int), cudaMemcpyDeviceToHost); // memcpyAsync #2

  printf("[after memcpyAsync #2] result: %d\n", result);

  printf("[before deviceSynchronize #1] result: %d\n", result);

  cudaDeviceSynchronize(); // deviceSynchronize #1

  printf("[after deviceSynchronize #1] result: %d\n", result);

  printf("### end FLOW 3 ###\n\n");
}

int main(void) {

  flow_1(); // deviceSynchronize + memcpy

  flow_2(); // memcpy

  flow_3(); // memcpyAsync

  /*

  helloCPU(3, 20000);

  helloGPU<<< 1, 1 >>>(4, 20000, dev_result); // potentially skipped or not in order, if deviceSynchronize #2 is missing

  //cudaDeviceSynchronize(); // deviceSynchronize #2

  helloCPU(5, 30000);

  helloCPU(6, 10000);

  helloGPU<<< 1, 1 >>>(7, 20000, dev_result); // potentially skipped or not in order, if deviceSychronize #3 is missing

  //cudaDeviceSynchronize(); // deviceSynchronize #3

  helloCPU(8, 10000);

  */

  return 0;
}
