/*
 * @Name: matrix_mul_nxn.cu
 * @Description: Multiplication of NxN integer matrices.
 * Each matrix is viewed as a single block of memory.
 * Blocks and threads are viewed as a 2D grid.
 * Custom matrix dimension and block size.
 *
 * @Author: Giacomo Marciani <gmarciani@acm.org>
 * @Institution: University of Rome Tor Vergata
 *
 * @Usage: matrix_mul_nxn matrixDim blockSize
 */

#include <stdio.h>
#include <math.h>
#include <../common/error.h>
#include <../common/random.h>
#include <../common/matrix.h>

__global__ void mul(int *a, int *b, int *c, int dim) {
  int iX = blockIdx.x * blockDim.x + threadIdx.x;
  int iY = blockIdx.y * blockDim.y + threadIdx.y;

  if (iX < dim && iY < dim) {
    int idx = iY * dim + iX;
    int val = 0;
    for (int k = 0; k < dim; k++) {
      val += a[iY * dim + k] * b[k * dim + iX];
    }

    c[idx] = val;
  }
}

int main(const int argc, const char **argv) {
  int *a, *b, *c;         // host copies of a, b, c
  int *dev_a, *dev_b, *dev_c; // device copies of a, b, c
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

  size = matrixDim * matrixDim * sizeof(int);

  // allocate host copy of a, b, c
  a = HANDLE_NULL((int*)malloc(size));
  b = HANDLE_NULL((int*)malloc(size));
  c = HANDLE_NULL((int*)malloc(size));

  // allocate device copy of a, b, c
  HANDLE_ERROR(cudaMalloc((void**)&dev_a, size));
  HANDLE_ERROR(cudaMalloc((void**)&dev_b, size));
  HANDLE_ERROR(cudaMalloc((void**)&dev_c, size));

  // fill a, b with random integers
  random_matrix_int(a, matrixDim, matrixDim)
  random_matrix_int(b, matrixDim, matrixDim)

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

  // launch mul() kernel
  mul<<< gridDim, blockDim >>>(dev_a, dev_b, dev_c, matrixDim);

  // copy device result back to host copy of c
  HANDLE_ERROR(cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost));

  // test result
  int *d = HANDLE_NULL((int*)malloc(size));
  matrix_mul(a, b, d, matrixDim, matrixDim, matrixDim);
  for (int y = 0; y < matrixDim; y++) {
    for (int x = 0; x < matrixDim; x++) {
      int idx = y * matrixDim + x;
      if (c[idx] != d[idx]) {
        fprintf(stderr, "Error: (%d,%d) expected %d, got %d\n",
        x, y, d[idx], c[idx]);
        break;
      }
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
