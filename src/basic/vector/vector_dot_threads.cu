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
#include <../common/error.h>
#include <../common/random.h>

#define VECTOR_DIM 512

__global__ void dot(int *a, int *b, int *c) {
  __shared__ int temp[N];

  temp[threadIdx.x] = a[threadIdx.x] * b[threadIdx.x];

  __syncthreads();

  if (0 == threadIdx.x) {
    int sum = 0;
    for (int i = N - 1; i >=0; i--) {
      sum += temp[i];
    }
  }

  *c = sum;
}

int main(void) {
  int *a, *b, c;              // host copies of a, b, c
  int *dev_a, *dev_b, *dev_c; // device copies of a, b, c
  int size = VECTOR_DIM * sizeof(int); // bytes for an array of VECTOR_DIM integers

  // allocate host copies of a, b, c
  a = HANDLE_NULL((int*)malloc(size));
  b = HANDLE_NULL((int*)malloc(size));
  c = HANDLE_NULL((int*)malloc(sizeof(int)));

  // allocate device copies of a, b, c
  HANDLE_ERROR(cudaMalloc((void**)&dev_a, size));
  HANDLE_ERROR(cudaMalloc((void**)&dev_b, size));
  HANDLE_ERROR(cudaMalloc((void**)&dev_c, sizeof(int)));

  // fill a and b with VECTOR_DIM random integers
  random_ints(a, VECTOR_DIM);
  random_ints(b, VECTOR_DIM);

  // copy inputs to device
  HANDLE_ERROR(cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice));

  // launch add() kernel
  dot<<< 1, VECTOR_DIM >>>(dev_a, dev_b, dev_c);

  // copy device result back to host copy of c
  HANDLE_ERROR(cudaMemcpy(c, dev_c, sizeof(int), cudaMemcpyDeviceToHost));

  // test result
  int d = 0;
  for(int i = 0; i < N; i++) {
    d += a[i] * b[i];
  }
  if (*c != d) {
    printf("Error: expected %d, got %d\n", d, *c);
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
