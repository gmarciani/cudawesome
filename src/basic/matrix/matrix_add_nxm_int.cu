/*
 * @Name: matrix_add_nxm_int.cu
 * @Description: Matrix (NxM) Integer Sum
 * Each matrix is viewed as a single block of memory.
 * Blocks and threads are viewed as a 2D grid.
 * Custom matrix dimensions and block size.
 *
 * @Author: Giacomo Marciani <gmarciani@acm.org>
 * @Institution: University of Rome Tor Vergata
 *
 * @Usage: matrix_add_nxm_int matrixDimX matrixDimY blockSize
 *
 * Default values:
 *  matrixDimX: 4096
 *  matrixDimY: 4096
 *  blockSize: 32
 */

#include <stdio.h>
#include <math.h>
#include "../../common/error.h"
#include "../../common/random.h"
#include "../../common/matrix.h"


#define MAX_BLOCK_SIZE 1024

__global__ void matrixAdd(const int *a, const int *b, int *c, const unsigned int dimX, const unsigned int dimY) {
  const unsigned int iX = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int iY = blockIdx.y * blockDim.y + threadIdx.y;

  if (iX >= dimX || iY >= dimY) return;

  const unsigned int pos = iY * dimX + iX;
  c[pos] = a[pos] + b[pos];
}

__host__ void gpuMatrixAdd(const int *a, const int *b, int *c, const unsigned int matrixDimX, const unsigned int matrixDimY, const dim3 gridDim, const dim3 blockDim) {
  int *dev_a, *dev_b, *dev_c; // device copies of a, b, c
  const unsigned int size = matrixDimX * matrixDimY * sizeof(int); // bytes for a, b, c

  cudaError_t err;

  // allocate device copies of a, b, c
  HANDLE_ERROR(cudaMalloc((void**)&dev_a, size));
  HANDLE_ERROR(cudaMalloc((void**)&dev_b, size));
  HANDLE_ERROR(cudaMalloc((void**)&dev_c, size));

  // copy inputs to device
  HANDLE_ERROR(cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice));

  // launch kernel matrixAdd()
  matrixAdd<<< gridDim, blockDim >>>(dev_a, dev_b, dev_c, matrixDimX, matrixDimY);

  err = cudaGetLastError();
  if ( err != cudaSuccess ) {
    printf("Error matrixAdd :: grid(%d,%d,%d) | block(%d,%d,%d) :: %s\n", gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, cudaGetErrorString(err));
  }

  // copy device result back to host copy of c
  HANDLE_ERROR(cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost));

  // free device
  HANDLE_ERROR(cudaFree(dev_a));
  HANDLE_ERROR(cudaFree(dev_b));
  HANDLE_ERROR(cudaFree(dev_c));
}

int main(const int argc, const char **argv) {
  int *a, *b, *c;             // host copies of a, b, c
  unsigned int size; // bytes for a, b, c
  unsigned int matrixDimX, matrixDimY; // matrix dimensions
  unsigned int blockSize; // block size
  cudaDeviceProp gpuInfo; // gpu properties

  // check arguments
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

  if (blockSize < 1 || blockSize > MAX_BLOCK_SIZE) {
    fprintf(stderr, "Error: blockSize expected >= 1 and <= %d, got %d\n", MAX_BLOCK_SIZE, blockSize);
    exit(1);
  }

  // grid settings
  dim3 gridDim(1, 1, 1);
  dim3 blockDim(1, 1, 1);

  if (matrixDimX >= matrixDimY) {
    blockDim.y = pow(blockSize, 1/2.);
    blockDim.x = blockSize / blockDim.y;
  } else {
    blockDim.x = pow(blockSize, 1/2.);
    blockDim.y = blockSize / blockDim.x;
  }

  gridDim.x = 1 + ((matrixDimX - 1) / blockDim.x);
  gridDim.y = 1 + ((matrixDimY - 1) / blockDim.y);

  size = matrixDimX * matrixDimY * sizeof(int);

  HANDLE_ERROR(cudaGetDeviceProperties(&gpuInfo, 0));

  printf("------------------------------------\n");
  printf("Matrix (NxM) Integer Sum\n");
  printf("------------------------------------\n");
  printf("Matrix Dimension: (%d, %d)\n", matrixDimX, matrixDimY);
  printf("Threads per block: %d\n", blockSize);
  printf("Grid Size: (%d, %d, %d) (max: (%d, %d, %d))\n",
    gridDim.x, gridDim.y, gridDim.z,
    gpuInfo.maxGridSize[0], gpuInfo.maxGridSize[1], gpuInfo.maxGridSize[2]);
  printf("Block Size: (%d, %d, %d) (max: (%d, %d, %d))\n",
    blockDim.x, blockDim.y, blockDim.z,
    gpuInfo.maxThreadsDim[0], gpuInfo.maxThreadsDim[1], gpuInfo.maxThreadsDim[2]);
  printf("-----------------------------------\n");

  // allocate host copies of a, b, c
  HANDLE_NULL(a = (int*)malloc(size));
  HANDLE_NULL(b = (int*)malloc(size));
  HANDLE_NULL(c = (int*)malloc(size));

  // fill a, b with random data
  random_matrix_int(a, matrixDimX, matrixDimY);
  random_matrix_int(b, matrixDimX, matrixDimY);

  // launch kernel matrixAdd()
  gpuMatrixAdd(a, b, c, matrixDimX, matrixDimY, gridDim, blockDim);

  // test result
  int *expected;
  HANDLE_NULL(expected = (int*)malloc(size));
  matrix_add_int(a, b, expected, matrixDimX, matrixDimY);
  const bool equal = matrix_equals_int(c, expected, matrixDimX, matrixDimY);
  if (!equal) {
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
