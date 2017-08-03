/*
 * @Name: matrix_add_nxm.cu
 * @Description: Addition of NxM integer matrices.
 * Each matrix is viewed as a single block of memory.
 * Blocks and threads are viewed as a 2D grid.
 * Custom matrix dimensions and block size.
 *
 * @Author: Giacomo Marciani <gmarciani@acm.org>
 * @Institution: University of Rome Tor Vergata
 *
 * @Usage: matrix_add_nxm matrixDimX matrixDimY blockSize
 */

#include <stdio.h>
#include <math.h>
#include "../../common/error.h"
#include "../../common/random.h"
#include "../../common/matrix.h"

__global__ void add(double *a, double *b, double*c, int dimX, int dimY) {
  int iX = blockIdx.x * blockDim.x + threadIdx.x;
  int iY = blockIdx.y * blockDim.y + threadIdx.y;

  if (iX < dimX && iY < dimY) {
    int idx = iY * dimX + iX;
    c[idx] = a[idx] + b[idx];
  }
}

int main(const int argc, const char **argv) {
  double *a, *b, *c;             // host copies of a, b, c
  double *dev_a, *dev_b, *dev_c; // device copies of a, b, c
  int size; // bytes for a, b, c
  int matrixDimX, matrixDimY; // matrix dimensions
  int gridSizeX, gridSizeY; // grid size
  int blockSize; // block size

  if (argc < 4) {
    fprintf(stderr, "Usage: %s matrixDimX matrixDimY blockSize\n", argv[0]);
    exit(1);
  }

  matrixDimX = atoi(argv[1]);
  matrixDimY = atoi(argv[2]);
  blockSize = atoi(argv[3]);

  if (matrixDimX < 1) {
    fprintf(stderr, "Error: matrixDimX expected >= 1, got %d\n", matrixDimX);
    exit(1);
  }

  if (matrixDimY < 1) {
    fprintf(stderr, "Error: matrixDimY expected >= 1, got %d\n", matrixDimY);
    exit(1);
  }

  if (blockSize < 1) {
    fprintf(stderr, "Error: blockSize expected >= 1, got %d\n", blockSize);
    exit(1);
  }

  size = matrixDimX * matrixDimY * sizeof(double);

  // allocate host copies of a, b, c
  HANDLE_NULL(a = (double*)malloc(size));
  HANDLE_NULL(b = (double*)malloc(size));
  HANDLE_NULL(c = (double*)malloc(size));

  // allocate device copies of a, b, c
  HANDLE_ERROR(cudaMalloc((void**)&dev_a, size));
  HANDLE_ERROR(cudaMalloc((void**)&dev_b, size));
  HANDLE_ERROR(cudaMalloc((void**)&dev_c, size));

  // fill a, b with random data
  random_matrix_double(a, matrixDimX, matrixDimY);
  random_matrix_double(b, matrixDimX, matrixDimY);

  // copy inputs to device
  HANDLE_ERROR(cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice));

  // grid settings
  dim3 gridDim, blockDim;
  gridSizeX = matrixDimX / blockSize;
  if (gridSizeX * blockSize < matrixDimX) {
     gridSizeX += 1;
  }
  gridSizeY = matrixDimY / blockSize;
  if (gridSizeY * blockSize < matrixDimY) {
     gridSizeY += 1;
  }
  blockDim.x = blockSize;
  blockDim.y = blockSize;
  gridDim.x = gridSizeX;
  gridDim.y = gridSizeY;

  // launch add() kernel
  add<<< gridDim, blockDim >>>(dev_a, dev_b, dev_c, matrixDimX, matrixDimY);

  // copy device result back to host copy of c
  HANDLE_ERROR(cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost));

  // test result
  double *d;
  HANDLE_NULL(d = (double*)malloc(size));
  matrix_add_double(a, b, d, matrixDimX, matrixDimY);
  if (!matrix_equals_double(c, d, matrixDimX, matrixDimY)) {
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
