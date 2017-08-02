/*
 * @Name: matrix_add_nxn.cu
 * @Description: Addition of NxN integer matrices.
 * Each matrix is viewed as a single block of memory.
 * Blocks and threads are viewed as a 2D grid.
 * Custom matrix dimension and block size.
 *
 * @Author: Giacomo Marciani <gmarciani@acm.org>
 * @Institution: University of Rome Tor Vergata
 *
 * @Usage: matrix_add_nxn matrixDim blockSize
 */

#include <stdio.h>
#include <math.h>
#include "../../common/error.h"
#include "../../common/random.h"
#include "../../common/matrix.h"

__global__ void add(double *a, double *b, double *c, int dim) {
  int iX = blockIdx.x * blockDim.x + threadIdx.x;
  int iY = blockIdx.y * blockDim.y + threadIdx.y;

  if (iX < dim && iY < dim) {
    int idx = iY * dim + iX;
    c[idx] = a[idx] + b[idx];
  }
}

int main(const int argc, const char **argv) {
  double *a, *b, *c;             // host copies of a, b, c
  double *dev_a, *dev_b, *dev_c; // device copies of a, b, c
  int size; // bytes for a matrix of matrixDim x matrixDim integers
  int matrixDim; // matrix dimension
  int gridSize; // grid size
  int blockSize; // block size

  if (argc < 3) {
    fprintf(stderr, "Usage: %s matrixDim blockSize\n", argv[0]);
    exit(1);
  }

  matrixDim = atoi(argv[1]);
  blockSize = atoi(argv[2]);

  if (matrixDim < 1) {
    fprintf(stderr, "Error: matrixDim expected >= 1, got %d\n", matrixDim);
    exit(1);
  }

  if (blockSize < 1) {
    fprintf(stderr, "Error: blockSize expected >= 1, got %d\n", blockSize);
    exit(1);
  }

  size = matrixDim * matrixDim * sizeof(double);

  // allocate host copies of a, b, c
  HANDLE_NULL(a = (double*)malloc(size));
  HANDLE_NULL(b = (double*)malloc(size));
  HANDLE_NULL(c = (double*)malloc(size));

  // allocate device copies of a, b, c
  HANDLE_ERROR(cudaMalloc((void**)&dev_a, size));
  HANDLE_ERROR(cudaMalloc((void**)&dev_b, size));
  HANDLE_ERROR(cudaMalloc((void**)&dev_c, size));

  // fill a, b with random integers
  random_matrix_int(a, matrixDim, matrixDim);
  random_matrix_int(b, matrixDim, matrixDim);

  // copy inputs to device
  HANDLE_ERROR(cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice));

  // grid settings
  dim3 gridDim, blockDim;
  gridSize = matrixDim / blockSize;
  if (gridSize * blockSize < matrixDim) {
     gridSize += 1;
  }
  blockDim.x = blockSize;
  blockDim.y = blockSize;
  gridDim.x = gridSize;
  gridDim.y = gridSize;

  // launch add() kernel
  add<<< gridDim, blockDim >>>(dev_a, dev_b, dev_c, matrixDim);

  // copy device result back to host copy of c
  HANDLE_ERROR(cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost));

  // test result
  double *d;
  HANDLE_NULL(d = (double*)malloc(size));
  matrix_add(a, b, d, matrixDim, matrixDim);
  int i;
  for (i = 0; i < matrixDim * matrixDim; i++) {
    if (c[i] != d[i]) {
      fprintf(stderr, "Error: (%d,%d) expected %f, got %f\n",
      i % matrixDim, i - (i % matrixDim) / matrixDim, d[i], c[i]);
      break;
    }
  }
  if (i == matrixDim * matrixDim) {
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
