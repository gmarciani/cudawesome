/*
 * @Name: vector_dot_threads.cu
 * @Description: Integer vectors dot-product.
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

__global__ void dot(double *a, double *b, double *c) {
  __shared__ double temp[VECTOR_DIM];

  temp[threadIdx.x] = a[threadIdx.x] * b[threadIdx.x];

  __syncthreads();

  if (0 == threadIdx.x) {
    double sum = 0;
    for (int i = VECTOR_DIM - 1; i >= 0; i--) {
      sum += temp[i];
    }
    *c = sum;
  }
}

int main(void) {
  double *a, *b, *c;             // host copies of a, b, c
  double *dev_a, *dev_b, *dev_c; // device copies of a, b, c
  int size = VECTOR_DIM * sizeof(double); // bytes for a, b
  int size_c = sizeof(double); // bytes for c

  // allocate host copies of a, b, c
  HANDLE_NULL(a = (double*)malloc(size));
  HANDLE_NULL(b = (double*)malloc(size));
  HANDLE_NULL(c = (double*)malloc(size_c));

  // allocate device copies of a, b, c
  HANDLE_ERROR(cudaMalloc((void**)&dev_a, size));
  HANDLE_ERROR(cudaMalloc((void**)&dev_b, size));
  HANDLE_ERROR(cudaMalloc((void**)&dev_c, size_c));

  // fill a and b with random data
  random_vector_double(a, VECTOR_DIM);
  random_vector_double(b, VECTOR_DIM);

  // copy inputs to device
  HANDLE_ERROR(cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice));

  // launch dot() kernel
  dot<<< 1, VECTOR_DIM >>>(dev_a, dev_b, dev_c);

  // copy device result back to host copy of c
  HANDLE_ERROR(cudaMemcpy(c, dev_c, size_c, cudaMemcpyDeviceToHost));

  // test result
  double d;
  vector_dot_double(a, b, &d, VECTOR_DIM);
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
