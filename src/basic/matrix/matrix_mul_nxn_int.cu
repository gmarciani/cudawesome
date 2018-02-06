/*
 * @Name: matrix_mul_nxn_int.cu
 * @Description: Matrix (NxN) Integer Product.
 * Each matrix is viewed as a single block of memory.
 * Blocks and threads are viewed as a 2D grid.
 * Custom matrix dimension and block size.
 *
 * @Author: Giacomo Marciani <gmarciani@acm.org>
 * @Institution: University of Rome Tor Vergata
 *
 * @Usage: matrix_mul_nxn_int matrixDim blockSize
 *
 * Default values:
 *  matrixDim: 4096
 *  blockSize: 32
 */

#include <stdio.h>
#include <math.h>
#include "../../common/error.h"
#include "../../common/random.h"
#include "../../common/matrix.h"
#include "../../common/mathutil.h"

__global__ void matrixMul(const int *a, const int *b, int *c, const unsigned int dim) {
  const unsigned int iX = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int iY = blockIdx.y * blockDim.y + threadIdx.y;

  if (iX >= dim || iY >= dim) return;

  const unsigned int pos = iY * dim + iX;

  int val = 0;
  for (unsigned int k = 0; k < dim; k++) {
    val += a[iY * dim + k] * b[k * dim + iX];
  }

  c[pos] = val;
}

__host__ void gpuMatrixMul(const int *a, const int *b, int *c, const unsigned int matrixDim, const dim3 gridDim, const dim3 blockDim) {
  int *dev_a, *dev_b, *dev_c; // device copies of a, b, c
  const unsigned int size = matrixDim * matrixDim * sizeof(int); // bytes for a, b, c

  // allocate device copy of a, b, c
  HANDLE_ERROR(cudaMalloc((void**)&dev_a, size));
  HANDLE_ERROR(cudaMalloc((void**)&dev_b, size));
  HANDLE_ERROR(cudaMalloc((void**)&dev_c, size));

  // copy inputs to device
  HANDLE_ERROR(cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice));

  // launch mul() kernel
  matrixMul<<< gridDim, blockDim >>>(dev_a, dev_b, dev_c, matrixDim);

  // copy device result back to host copy of c
  HANDLE_ERROR(cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost));

  // free device
  HANDLE_ERROR(cudaFree(dev_a));
  HANDLE_ERROR(cudaFree(dev_b));
  HANDLE_ERROR(cudaFree(dev_c));
}

int main(const int argc, const char **argv) {
  int *a, *b, *c;         // host copies of a, b, c
  unsigned int size; // bytes for a, b, c
  unsigned int matrixDim; // matrices dimensions
  unsigned int gridSizeX, gridSizeY; // grid size
  unsigned int blockSize; // block size
  cudaDeviceProp gpuInfo; // gpu properties

  // check arguments
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

  if (!IS_POWER_OF_2(blockSize)) {
    fprintf(stderr, "Error: blockSize expected as power of 2, got %d\n", blockSize);
    exit(1);
  }

  // grid settings
  gridSizeX = matrixDim / blockSize;
  if (gridSizeX * blockSize < matrixDim) {
     gridSizeX += 1;
  }
  gridSizeY = matrixDim / blockSize;
  if (gridSizeY * blockSize < matrixDim) {
     gridSizeY += 1;
  }
  dim3 gridDim(gridSizeX, gridSizeY);
  dim3 blockDim(blockSize, blockSize);

  size = matrixDim * matrixDim * sizeof(int);

  HANDLE_ERROR(cudaGetDeviceProperties(&gpuInfo, 0));

  printf("------------------------------------\n");
  printf("Matrix (NxM) Integer Product\n");
  printf("------------------------------------\n");
  printf("Matrix Dimension (A): (%d, %d)\n", matrixDim, matrixDim);
  printf("Matrix Dimension (B): (%d, %d)\n", matrixDim, matrixDim);
  printf("Grid Size: (%d, %d, %d) (max: (%d, %d, %d))\n",
    gridDim.x, gridDim.y, gridDim.z,
    gpuInfo.maxGridSize[0], gpuInfo.maxGridSize[1], gpuInfo.maxGridSize[2]);
  printf("Block Size: (%d, %d, %d) (max: (%d, %d, %d))\n",
    blockDim.x, blockDim.y, blockDim.z,
    gpuInfo.maxThreadsDim[0], gpuInfo.maxThreadsDim[1], gpuInfo.maxThreadsDim[2]);
  printf("-----------------------------------\n");

  // allocate host copy of a, b, c
  HANDLE_NULL(a = (int*)malloc(size));
  HANDLE_NULL(b = (int*)malloc(size));
  HANDLE_NULL(c = (int*)malloc(size));

  // fill a, b with random data
  random_matrix_int(a, matrixDim, matrixDim);
  random_matrix_int(b, matrixDim, matrixDim);

  // launch kernel matrixMul()
  gpuMatrixMul(a, b, c, matrixDim, gridDim, blockDim);

  // test result
  int *expected;
  HANDLE_NULL(expected = (int*)malloc(size));
  matrix_mul_int(a, b, expected, matrixDim, matrixDim, matrixDim);
  const bool correct = matrix_equals_int(c, expected, matrixDim, matrixDim);
  if (!correct) {
    fprintf(stderr, "Error\n");
  } else {
    printf("Correct\n");
  }

  // free host
  free(a);
  free(b);
  free(c);
  free(expected);

  return 0;
}
