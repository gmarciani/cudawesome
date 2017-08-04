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
#include "../../common/error.h"
#include "../../common/random.h"
#include "../../common/vector.h"

#ifdef DOUBLE
#define REAL double
#else
#define REAL float
#endif

#define VECTOR_DIM 512
#define BLOCK_SIZE 16

__global__ void add(const REAL *a, const REAL *b, REAL *c) {
  const unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;
  c[pos] = a[pos] + b[pos];
}

int main(void) {
  REAL *a, *b, *c;             // host copies of a, b, c
  REAL *dev_a, *dev_b, *dev_c; // device copies of a, b, c
  const unsigned int size = VECTOR_DIM * sizeof(REAL); // bytes for a, b, c

  #ifdef DOUBLE
  printf("Double precision\n");
  #else
  printf("Single precision\n");
  #endif

  // allocate host copies of a, b, c
  HANDLE_NULL(a = (REAL*)malloc(size));
  HANDLE_NULL(b = (REAL*)malloc(size));
  HANDLE_NULL(c = (REAL*)malloc(size));

  // allocate device copies of a, b, c
  HANDLE_ERROR(cudaMalloc((void**)&dev_a, size));
  HANDLE_ERROR(cudaMalloc((void**)&dev_b, size));
  HANDLE_ERROR(cudaMalloc((void**)&dev_c, size));

  // fill a, b with random data
  #ifdef DOUBLE
  random_vector_double(a, VECTOR_DIM);
  random_vector_double(b, VECTOR_DIM);
  #else
  random_vector_float(a, VECTOR_DIM);
  random_vector_float(b, VECTOR_DIM);
  #endif

  // copy inputs to device
  HANDLE_ERROR(cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice));

  // launch add() kernel
  add<<< VECTOR_DIM / BLOCK_SIZE, BLOCK_SIZE >>>(dev_a, dev_b, dev_c);

  // copy device result back to host copy of c
  HANDLE_ERROR(cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost));

  // test result
  REAL *d;
  HANDLE_NULL(d = (REAL*)malloc(size));
  #ifdef DOUBLE
  vector_add_double(a, b, d, VECTOR_DIM);
  const bool equal = vector_equals_double(c, d, VECTOR_DIM);
  #else
  vector_add_float(a, b, d, VECTOR_DIM);
  const bool equal = vector_equals_float(c, d, VECTOR_DIM);
  #endif
  if (!equal) {
    fprintf(stderr, "Error\n");
  } else {
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
