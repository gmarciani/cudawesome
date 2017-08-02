/*
 * @Name: event_matrix_mul.cu
 * @Description: Multiplication of NxN integer matrices.
 * Each matrix is viewed as a single block of memory.
 * One block, multiple threads.
 * Elapsed time recorded with CUDA Event API.
 *
 * @Author: Giacomo Marciani <gmarciani@acm.org>
 * @Institution: University of Rome Tor Vergata
 */

#include <stdio.h>
#include <math.h>
#include "../common/error.h"
#include "../common/random.h"
#include "../common/matrix.h"

#define MAX_MATRIX_DIM 16

__global__ void mul(int *a, int *b, int *c, int dim) {
  int idx = threadIdx.y * dim + threadIdx.x;

  int val = 0;
  for(int k = 0; k < dim; k++) {
    val += a[threadIdx.y * dim + k] * b[k * dim + threadIdx.x];
  }

  c[idx] = val;
}

int main(void) {
  int *a, *b, *c, *d;         // host copy of a
  int *dev_a, *dev_b, *dev_c; // device copy of a
  int matrixDim;              // matrix dimension
  int size; // bytes for a matrix of matrixDim x matrixDim integers

  cudaEvent_t start, stop; // events for elapsed time
  float elapsed; // elapsed time

  if(argc < 2) {
    fprintf(stderr, "Usage: %s MatrixDimension\n", argv[0]);
    exit(1);
  }

  matrixDim = atoi(argv[1]);

  if(matrixDim < 1 || matrixDim > MAX_MATRIX_DIM) {
    fprintf(stderr, "Error: matrixDim expected in [1,%d], got %d\n", MAX_MATRIX_DIM, matrixDim);
    exit(1);
  }

  size = matrixDim * matrixDim * sizeof(int);

  // allocate host copy of a, b, c, d
  a = HANDLE_NULL((int*)malloc(size));
  b = HANDLE_NULL((int*)malloc(size));
  c = HANDLE_NULL((int*)malloc(size));
  d = HANDLE_NULL((int*)malloc(size));

  // allocate device copy of a, b, c
  HANDLE_ERROR(cudaMalloc((void**)&dev_a, size));
  HANDLE_ERROR(cudaMalloc((void**)&dev_b, size));
  HANDLE_ERROR(cudaMalloc((void**)&dev_c, size));

  // fill a, b with random integers
  random_matrix_int(a, matrixDim, matrixDim)
  random_matrix_int(b, matrixDim, matrixDim)

  // grid settings
  dim3 gridDim, blockDim;
  blockDim.x = matrixDim;
  blockDim.y = matrixDim;
  gridDim.x = 1;
  gridDim.y = 1;

  // allocate events start, stop
  HANDLE_ERROR(cudaEventCreate(&start));
  HANDLE_ERROR(cudaEventCreate(&stop));

  cudaEventRecord(start, 0);

  // launch mul() kernel
  mul<<< gridDim, blockDim >>>(dev_a, dev_b, dev_c, MATRIX_DIM);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed, start, stop);

  // copy device result back to host copy of c
  HANDLE_ERROR(cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost));

  // free host
  free(a);
  free(b);
  free(c);

  // free device
  HANDLE_ERROR(cudaFree(dev_a));
  HANDLE_ERROR(cudaFree(dev_b));
  HANDLE_ERROR(cudaFree(dev_c));

  // free events
  HANDLE_ERROR(cudaEventDestroy(start));
  HANDLE_ERROR(cudaEventDestroy(stop));

  printf("Kernel execution time: %f ms\n", elapsed);

  return 0;
}
