/*
 * @Name: matrix_mul_nxm.cu
 * @Description: Multiplication of NxN integer matrices.
 * Each matrix is viewed as a single block of memory.
 * Blocks and threads are viewed as a 2D grid.
 * Custom matrix dimension and block size.
 *
 * @Author: Giacomo Marciani <gmarciani@acm.org>
 * @Institution: University of Rome Tor Vergata
 *
 * @Usage: matrix_mul_nxm matrixDimX1 matrixDimY1 matrixDimX2 matrixDimY2 blockSize
 */

#include <stdio.h>
#include <math.h>
#include "../../common/error.h"
#include "../../common/random.h"
#include "../../common/matrix.h"

__global__ void mul(double *a, double *b, double *c, int dimX1, int dimY1, int dimX2) {
  int iX = blockIdx.x * blockDim.x + threadIdx.x;
  int iY = blockIdx.y * blockDim.y + threadIdx.y;

  if (iX < dimX2 && iY < dimY1) {
    int idx = iY * dimX2 + iX;
    double val = 0;
    for (int k = 0; k < dimX1; k++) {
      val += a[iY * dimX1 + k] * b[k * dimX2 + iX];
    }

    c[idx] = val;
  }
}

int main(const int argc, const char **argv) {
  double *a, *b, *c;         // host copies of a, b, c
  double *dev_a, *dev_b, *dev_c; // device copies of a, b, c
  int size_a, size_b, size_c; // bytes for a, b, c
  int matrixDimX1, matrixDimY1, matrixDimX2, matrixDimY2; // matrices dimensions
  int gridSizeX, gridSizeY; // grid size
  int blockSize; // block size

  if (argc < 6) {
    fprintf(stderr, "Usage: %s matrixDimX1 matrixDimY1 matrixDimX2 matrixDimY2 blockSize\n", argv[0]);
    exit(1);
  }

  matrixDimX1 = atoi(argv[1]);
  matrixDimY1 = atoi(argv[2]);
  matrixDimX2 = atoi(argv[3]);
  matrixDimY2 = atoi(argv[4]);
  blockSize = atoi(argv[5]);

  if (matrixDimX1 < 1) {
    fprintf(stderr, "Error: matrixDimX1 expected >= 1, got %d\n", matrixDimX1);
    exit(1);
  }

  if (matrixDimY1 < 1) {
    fprintf(stderr, "Error: matrixDimY1 expected >= 1, got %d\n", matrixDimY1);
    exit(1);
  }

  if (matrixDimX2 < 1) {
    fprintf(stderr, "Error: matrixDimX2 expected >= 1, got %d\n", matrixDimX2);
    exit(1);
  }

  if (matrixDimY2 != matrixDimX1) {
    fprintf(stderr, "Error: matrixDimY2 expected = matrixDimX1 (%d), got %d\n", matrixDimX1, matrixDimY2);
    exit(1);
  }

  if (blockSize < 1) {
    fprintf(stderr, "Error: blockSize expected >= 1, got %d\n", blockSize);
    exit(1);
  }

  size_a = matrixDimX1 * matrixDimY1 * sizeof(double);
  size_b = matrixDimX2 * matrixDimY2 * sizeof(double);
  size_c = matrixDimY1 * matrixDimX2 * sizeof(double);

  // allocate host copy of a, b, c
  HANDLE_NULL(a = (double*)malloc(size_a));
  HANDLE_NULL(b = (double*)malloc(size_b));
  HANDLE_NULL(c = (double*)malloc(size_c));

  // allocate device copy of a, b, c
  HANDLE_ERROR(cudaMalloc((void**)&dev_a, size_a));
  HANDLE_ERROR(cudaMalloc((void**)&dev_b, size_b));
  HANDLE_ERROR(cudaMalloc((void**)&dev_c, size_c));

  // fill a, b with random data
  random_matrix_double(a, matrixDimX1, matrixDimY1);
  random_matrix_double(b, matrixDimX2, matrixDimY2);

  // copy inputs to device
  HANDLE_ERROR(cudaMemcpy(dev_a, a, size_a, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_b, b, size_b, cudaMemcpyHostToDevice));

  // grid settings
  dim3 gridDim, blockDim;
  int maxDimX = max(matrixDimX1, matrixDimX2);
  gridSizeX = maxDimX / blockSize;
  if (gridSizeX * blockSize < maxDimX) {
     gridSizeX += 1;
  }
  int maxDimY = max(matrixDimY1, matrixDimY2);
  gridSizeY = maxDimY / blockSize;
  if (gridSizeY * blockSize < maxDimY) {
     gridSizeY += 1;
  }
  blockDim.x = blockSize;
  blockDim.y = blockSize;
  gridDim.x = gridSizeX;
  gridDim.y = gridSizeY;

  // launch mul() kernel
  mul<<< gridDim, blockDim >>>(dev_a, dev_b, dev_c, matrixDimX1, matrixDimY1, matrixDimX2);

  // copy device result back to host copy of c
  HANDLE_ERROR(cudaMemcpy(c, dev_c, size_c, cudaMemcpyDeviceToHost));

  // test result
  double *d;
  HANDLE_NULL(d = (double*)malloc(size_c));
  matrix_mul_double(a, b, d, matrixDimX1, matrixDimY1, matrixDimX2);
  if (!matrix_equals_double(c, d, matrixDimX2, matrixDimY1)) {
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
