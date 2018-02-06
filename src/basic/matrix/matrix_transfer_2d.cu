/*
 * @Name: matrix_transfer_2d.cu
 * @Description: 2D Matrix (NxM) Floating-Point Transfer.
 *
 * @Author: Giacomo Marciani <gmarciani@acm.org>
 * @Institution: University of Rome Tor Vergata
 *
 * @Usage: matrix_transfer_2d matrixRows matrixCols blockSize
 *
 * Default values:
 *  matrixRows: 4096
 *  matrixCols: 4096
 *  blockSize: 32
 */

#include <stdio.h>
#include <math.h>
#include "../../common/error.h"
#include "../../common/random.h"
#include "../../common/matrix.h"

#ifdef DOUBLE
#define REAL double
#else
#define REAL float
#endif

__global__ void matrixCopy(REAL *a, REAL *b, const unsigned int matrixRows, const unsigned int matrixCols) {
  const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= matrixRows || col >= matrixCols) return;

  const unsigned int pos = row * matrixCols + col;

  b[pos] = a[pos];
}

__host__ void gpuMatrixCopy(REAL **a, REAL **b, const unsigned int matrixRows, const unsigned int matrixCols, const dim3 gridDim, const dim3 blockDim) {
  REAL *dev_a = NULL; // device copies of a, b
  REAL *dev_b = NULL; // device copies of a, b
  const size_t size = matrixRows * matrixCols * sizeof(REAL); // bytes for a, b
  const size_t sizeX = matrixCols * sizeof(REAL); // bytes for a, b (dimension X)
  unsigned int r; // indices

  // allocate device copies of a, b
  HANDLE_ERROR(cudaMalloc((void**)&dev_a, size));
  HANDLE_ERROR(cudaMalloc((void**)&dev_b, size));

  // copy inputs to device
  for (r = 0; r < matrixRows; r++) {
    HANDLE_ERROR(cudaMemcpy((void*)(dev_a + (r * matrixCols)), (const void*)a[r], sizeX, cudaMemcpyHostToDevice));
  }

  // launch kernel matrixCopy()
  matrixCopy<<< gridDim, blockDim >>>(dev_a, dev_b, matrixRows, matrixCols);

  // copy device result back to host copy of b
  for (r = 0; r < matrixRows; r++) {
    HANDLE_ERROR(cudaMemcpy((void*)b[r], (const void*)(dev_b + (r * matrixCols)), sizeX, cudaMemcpyDeviceToHost));
  }

  // free device
  HANDLE_ERROR(cudaFree(dev_a));
  HANDLE_ERROR(cudaFree(dev_b));
}

int main(const int argc, const char **argv) {
  REAL **a, **b = NULL; // host copies of a, b
  unsigned int sizeX, sizeY; // bytes for a, b
  unsigned int matrixRows, matrixCols; // matrix dimensions
  unsigned int gridSizeX, gridSizeY; // grid size
  unsigned int blockSize; // block size
  cudaDeviceProp gpuInfo; // gpu properties
  unsigned int r, c; // indices

  // check arguments
  if (argc < 4) {
    fprintf(stderr, "Usage: %s matrixRows matrixCols blockSize\n", argv[0]);
    exit(1);
  }

  matrixRows = atoi(argv[1]);
  matrixCols = atoi(argv[2]);
  blockSize = atoi(argv[3]);

  if (matrixRows < 1) {
    fprintf(stderr, "Error: matrixRows expected >= 1, got %d\n", matrixRows);
    exit(1);
  }

  if (matrixCols < 1) {
    fprintf(stderr, "Error: matrixCols expected >= 1, got %d\n", matrixCols);
    exit(1);
  }

  if (blockSize < 1) {
    fprintf(stderr, "Error: blockSize expected >= 1, got %d\n", blockSize);
    exit(1);
  }

  // grid settings
  gridSizeX = matrixCols / blockSize;
  if (gridSizeX * blockSize < matrixCols) {
     gridSizeX += 1;
  }
  gridSizeY = matrixRows / blockSize;
  if (gridSizeY * blockSize < matrixRows) {
     gridSizeY += 1;
  }
  dim3 gridDim(gridSizeX, gridSizeY);
  dim3 blockDim(blockSize, blockSize);

  sizeY = matrixRows * sizeof(REAL*);
  sizeX = matrixCols * sizeof(REAL);

  HANDLE_ERROR(cudaGetDeviceProperties(&gpuInfo, 0));

  printf("---------------------------------------\n");
  printf("2D Matrix (NxM) Floating-Point Transfer\n");
  printf("---------------------------------------\n");
  #ifdef DOUBLE
  printf("FP Precision: Double\n");
  #else
  printf("FP Precision: Single\n");
  #endif
  printf("Matrix Dimension: (%d, %d)\n", matrixRows, matrixCols);
  printf("Grid Size: (%d, %d, %d) (max: (%d, %d, %d))\n",
    gridDim.x, gridDim.y, gridDim.z,
    gpuInfo.maxGridSize[0], gpuInfo.maxGridSize[1], gpuInfo.maxGridSize[2]);
  printf("Block Size: (%d, %d, %d) (max: (%d, %d, %d))\n",
    blockDim.x, blockDim.y, blockDim.z,
    gpuInfo.maxThreadsDim[0], gpuInfo.maxThreadsDim[1], gpuInfo.maxThreadsDim[2]);
  printf("-----------------------------------\n");

  // allocate host copies of a, b
  HANDLE_NULL(a = (REAL**)malloc(sizeY));
  for (r = 0; r < matrixRows; r++) {
    HANDLE_NULL(a[r] = (REAL*)malloc(sizeX));
  }
  HANDLE_NULL(b = (REAL**)malloc(sizeY));
  for (r = 0; r < matrixRows; r++) {
    HANDLE_NULL(b[r] = (REAL*)malloc(sizeX));
  }

  // fill a with random data
  #ifdef DOUBLE
  random_matrix_double_2d(a, matrixRows, matrixCols);
  #else
  random_matrix_float_2d(a, matrixRows, matrixCols);
  #endif

  // launch kernel matrixCopy()
  gpuMatrixCopy(a, b, matrixRows, matrixCols, gridDim, blockDim);

  // test result
  bool err = false;
  for (r = 0; r < matrixRows && !err; r++) {
    for (c = 0; c < matrixCols && !err; c++) {
      if (a[r][c] != b[r][c]) {
        err = true;
        break;
      }
    }
  }

  if (err) {
    fprintf(stderr, "Error\n");
  } else {
    printf("Correct!\n");
  }

  // free host
  for (r = 0; r < matrixRows; r++) {
    free(a[r]);
  }
  free(a);

  for (r = 0; r < matrixRows; r++) {
    free(b[r]);
  }
  free(b);

  return 0;
}
