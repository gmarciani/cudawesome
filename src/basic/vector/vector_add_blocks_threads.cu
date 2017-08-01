/*
 * @Name: vector_add_blocks_threads.cu
 * @Description: Addition of two integer vectors.
 * Multiple blocks, multiple threads per block.
 *
 * @Author: Giacomo Marciani <gmarciani@acm.org>
 * @Institution: University of Rome Tor Vergata
 */

#include <stdio.h>
#include <math.h>
#include <../common/error.h>
#include <../common/random.h>
#include <../common/vector.h>

#define VECTOR_DIM 512
#define BLOCK_SIZE 16

__global__ void add(int *a, int *b, int *c) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  c[idx] = a[idx] + b[idx];
}

int main(void) {
  int *a, *b, *c;             // host copies of a, b, c
  int *dev_a, *dev_b, *dev_c; // device copies of a, b, c
  int size = VECTOR_DIM * sizeof(int); // bytes for an array of VECTOR_DIM integers

  // allocate host copies of a, b, c
  a = HANDLE_NULL((int*)malloc(size));
  b = HANDLE_NULL((int*)malloc(size));
  c = HANDLE_NULL((int*)malloc(size));

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
  add<<< VECTOR_DIM / BLOCK_SIZE, BLOCK_SIZE >>>(dev_a, dev_b, dev_c);

  // copy device result back to host copy of c
  HANDLE_ERROR(cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost));

  // test result
  int *d = HANDLE_NULL((int*)malloc(size));
  vector_add(a, b, d, vectorDim);
  for (int i = 0; i < vectorDim; i++) {
    if (c[i] != d[i]) {
      printf("Error: [%d] expected %d, got %d\n", i, d, c[i]);
      break;
    }
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
