/*
 * @Name: vector_add_threads.cu
 * @Description: Addition of two integer vectors.
 * One block, multiple threads per block.
 *
 * @Author: Giacomo Marciani <gmarciani@acm.org>
 * @Institution: University of Rome Tor Vergata
 */

#include <stdio.h>
#include <math.h>
#include "../../common/error.h"
#include "../../common/random.h"
#include "../../common/vector.h"

#define VECTOR_DIM 512

__global__ void add(int *a, int *b, int *c) {
  c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}

int main(void) {
  int *a, *b, *c;             // host copies of a, b, c
  int *dev_a, *dev_b, *dev_c; // device copies of a, b, c
  int size = VECTOR_DIM * sizeof(int); // bytes for an array of VECTOR_DIM integers

  // allocate host copies of a, b, c
  HANDLE_NULL(a = (int*)malloc(size));
  HANDLE_NULL(b = (int*)malloc(size));
  HANDLE_NULL(c = (int*)malloc(size));

  // allocate device copies of a, b, c
  HANDLE_ERROR(cudaMalloc((void**)&dev_a, size));
  HANDLE_ERROR(cudaMalloc((void**)&dev_b, size));
  HANDLE_ERROR(cudaMalloc((void**)&dev_c, size));

  // fill a and b with VECTOR_DIM random integers
  random_ints(a, VECTOR_DIM);
  random_ints(b, VECTOR_DIM);

  // copy inputs to device
  HANDLE_ERROR(cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice));

  // launch add() kernel
  add<<< 1, VECTOR_DIM >>>(dev_a, dev_b, dev_c);

  // copy device result back to host copy of c
  HANDLE_ERROR(cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost));

  // test result
  int *d;
  HANDLE_NULL(d = (int*)malloc(size));
  vector_add(a, b, d, VECTOR_DIM);
  int i;
  for (i = 0; i < VECTOR_DIM; i++) {
    if (c[i] != d[i]) {
      fprintf(stderr, "Error: [%d] expected %d, got %d\n", i, d, c[i]);
      break;
    }
  }
  if (i == VECTOR_DIM) {
    printf("Correct\n");
  }

  // free host
  free(a);
  free(b);
  free(c);
  free(d);

  // free device
  HANDLE_ERROR(cudaFree(dev_a));
  HANDLE_ERROR(cudaFree(dev_b));
  HANDLE_ERROR(cudaFree(dev_c));

  return 0;
}
