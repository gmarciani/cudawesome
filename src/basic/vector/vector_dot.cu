/*
 * @Name: vector_dot.cu
 * @Description: Integer vectors dot-product.
 * Custom vector dimension and block size.
 *
 * @Author: Giacomo Marciani <gmarciani@acm.org>
 * @Institution: University of Rome Tor Vergata
 *
 * @Usage: vector_dot vectorDimension blockSize
 */

#include <stdio.h>
#include <math.h>
#include "../../common/error.h"
#include "../../common/random.h"
#include "../../common/vector.h"

__global__ void dot(int *a, int *b, int *c, int dim) {
  extern __shared__ int temp[];

  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx < dim) {
    temp[idx] = a[idx] * b[idx];

    __syncthreads();

    if (0 == threadIdx.x) {
      int sum = 0;
      for (int i = dim - 1; i >= 0; i--) {
        sum += temp[i];
      }
      atomicAdd(c, sum);
    }
  }
}

int main(const int argc, const char **argv) {
  int *a, *b, *c;             // host copies of a, b, c
  int *dev_a, *dev_b, *dev_c; // device copies of a, b, c
  int size; // bytes for an array of integers
  int vectorDim; // vector dimension
  int gridSize;  // grid size
  int blockSize; // block size

  if (argc < 3) {
    fprintf(stderr, "Usage: %s vectorDim blockSize\n", argv[0]);
    exit(1);
  }

  vectorDim = atoi(argv[1]);
  blockSize = atoi(argv[2]);

  if (vectorDim < 1) {
    fprintf(stderr, "Error: vectorDim expected >= 1, got %d\n", vectorDim);
    exit(1);
  }

  if (blockSize < 1) {
    fprintf(stderr, "Error: blockSize expected >= 1, got %d\n", blockSize);
    exit(1);
  }

  size = vectorDim * sizeof(int);

  // allocate host copies of a, b, c
  HANDLE_NULL(a = (int*)malloc(size));
  HANDLE_NULL(b = (int*)malloc(size));
  HANDLE_NULL(c = (int*)malloc(sizeof(int)));

  // allocate device copies of a, b, c
  HANDLE_ERROR(cudaMalloc((void**)&dev_a, size));
  HANDLE_ERROR(cudaMalloc((void**)&dev_b, size));
  HANDLE_ERROR(cudaMalloc((void**)&dev_c, sizeof(int)));

  // fill a and b with vectorDim random integers
  random_ints(a, vectorDim);
  random_ints(b, vectorDim);

  // copy inputs to device
  HANDLE_ERROR(cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemset(dev_c, 0, sizeof(int)));

  // grid settings
  gridSize = vectorDim / blockSize;
  if (gridSize * blockSize < vectorDim) {
    gridSize += 1;
  }

  // shared memory settings
  unsigned int sharedMemSize = (unsigned int) vectorDim * sizeof(int);

  // launch dot() kernel
  dot<<< gridSize, blockSize, sharedMemSize >>>(dev_a, dev_b, dev_c, vectorDim);

  // copy device result back to host copy of c
  HANDLE_ERROR(cudaMemcpy(c, dev_c, sizeof(int), cudaMemcpyDeviceToHost));

  // test result
  int d;
  vector_dot(a, b, &d, vectorDim);
  if (*c != d) {
    fprintf(stderr, "Error: expected %d, got %d\n", d, *c);
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
