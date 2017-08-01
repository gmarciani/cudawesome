/*
 * @Name: addint_ptr.cu
 * @Description: Integer addition with CUDA.
 * Arguments passed as pointers.
 * One block, one thread.
 *
 * @Author: Giacomo Marciani <gmarciani@acm.org>
 */

#include <stdio.h>
#include <math.h>
#include <../common/error.h>

__global__ void add(int *a, int *b, int *c) {
  *c = *a + *b;
}

int main(void) {
  int a, b, c;                // host copies of a, b, c
  int *dev_a, *dev_b, *dev_c; // device copies of a, b, c
  int size = sizeof(int);     // size of integers

  // allocate device copies of a, b, c
  cudaMalloc((void**)&dev_a, size);
  cudaMalloc((void**)&dev_b, size);
  cudaMalloc((void**)&dev_c, size);

  a = 2;
  b = 7;

  // copy inputs to device
  cudaMemcpy(dev_a, &a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, &b, size, cudaMemcpyHostToDevice);

  // launch an add() kernel with N threads
  add<<<1, 1>>>(dev_a, dev_b, dev_c);

  // copy device result back to host copy of c
  cudaMemcpy(&c, dev_c, size, cudaMemcpyDeviceToHost);

  printf("2 + 7 = %d\n", c);

  HANDLE_ERROR(cudaFree(dev_a));
  HANDLE_ERROR(cudaFree(dev_b));
  HANDLE_ERROR(cudaFree(dev_c));

  return 0;
}
