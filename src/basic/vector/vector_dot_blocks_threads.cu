/*
 * @Name: vector_dot_blocks_threads.cu
 * @Description: Integer vectors dot-product.
 * Multiple blocks, multiple threads per block.
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
#define BLOCK_SIZE 16

__global__ void dot(int *a, int *b, int *c) {
  __shared__ int temp[VECTOR_DIM];

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  temp[idx] = a[idx] * b[idx];

  __syncthreads();

  if (0 == threadIdx.x) {
    int sum = 0;
    for (int i = VECTOR_DIM - 1; i >= 0; i--) {
      sum += temp[i];
    }
    atomicAdd(c, sum);
  }
}

int main(void) {
  int *a, *b, *c;             // host copies of a, b, c
  int *dev_a, *dev_b, *dev_c; // device copies of a, b, c
  int size = VECTOR_DIM * sizeof(int); // bytes for a, b
  int size_c = sizeof(int); // bytes for c

  // allocate host copies of a, b, c
  HANDLE_NULL(a = (int*)malloc(size));
  HANDLE_NULL(b = (int*)malloc(size));
  HANDLE_NULL(c = (int*)malloc(size_c));

  // allocate device copies of a, b, c
  HANDLE_ERROR(cudaMalloc((void**)&dev_a, size));
  HANDLE_ERROR(cudaMalloc((void**)&dev_b, size));
  HANDLE_ERROR(cudaMalloc((void**)&dev_c, size_c));

  // fill a and b with random data
  random_vector_int(a, VECTOR_DIM);
  random_vector_int(b, VECTOR_DIM);

  // copy inputs to device
  HANDLE_ERROR(cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemset(dev_c, 0, size_c));

  // launch dot() kernel
  dot<<< VECTOR_DIM / BLOCK_SIZE, BLOCK_SIZE >>>(dev_a, dev_b, dev_c);

  // copy device result back to host copy of c
  HANDLE_ERROR(cudaMemcpy(c, dev_c, size_c, cudaMemcpyDeviceToHost));

  // test result
  int d;
  vector_dot_int(a, b, &d, VECTOR_DIM);
  if (*c != d) {
    fprintf(stderr, "Error: expected %f, got %f\n", d, *c);
  } else {
    printf("Correct\n");
  }

  // free host
  free(a);
  free(b);
  free(c);

  // free device
  HANDLE_ERROR(cudaFree(dev_a));
  HANDLE_ERROR(cudaFree(dev_b));
  HANDLE_ERROR(cudaFree(dev_c));

  return 0;
}
