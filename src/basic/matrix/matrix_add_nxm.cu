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

__global__ void add(int *a, int *b, int*c, int dimX, int dimY) {
  int iX = blockIdx.x * blockDim.x + threadIdx.x;
  int iY = blockIdx.y * blockDim.y + threadIdx.y;

  if (iX < dimX && iY < dimY) {
    int idx = iY * dim + iX;
    c[idx] = a[idx] + b[idx];
  }
}

int main(const int argc, const char **argv) {
  int *a, int *b, int *c;     // host copies of a, b, c
  int *dev_a, *dev_b, *dev_c; // device copies of a, b, c
  int size; // bytes for a matrix of matrixDimX x matrixDimY integers
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

  size = matrixDimX * matrixDimY * sizeof(int);

  // allocate host copies of a, b, c
  a = HANDLE_NULL((int*)malloc(size));
  b = HANDLE_NULL((int*)malloc(size));
  c = HANDLE_NULL((int*)malloc(size));

  // allocate device copies of a, b, c
  HANDLE_ERROR(cudaMalloc((void**)&dev_a, size));
  HANDLE_ERROR(cudaMalloc((void**)&dev_b, size));
  HANDLE_ERROR(cudaMalloc((void**)&dev_c, size));

  // fill a, b with random integers
  random_matrix_int(a, matrixDimX, matrixDimY)
  random_matrix_int(b, matrixDimX, matrixDimY)

  // grid settings
  dim3 gridDim, blockDim;
  gridsizeX = matrixDimX / blockSize;
  if (gridSizeX * blockSize < matrixDimX) {
     gridSizeX += 1;
  }
  gridsizeY = matrixDimY / blockSize;
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
  int *d = HANDLE_NULL((int*)malloc(size));
  matrix_add(a, b, d, matrixDimX, matrixDimY)
  for (int y = 0; y < matrixDimY; y++) {
    for (int x = 0; x < matrixDimX; x++) {
      int idx = y * matrixDimX + x;
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
