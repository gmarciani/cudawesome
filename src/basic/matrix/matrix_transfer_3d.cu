/*
 * @Name: matrix_transfer_2d.cu
 * @Description: 3D Matrix (NxMxZ) Floating-Point Transfer.
 *
 * @Author: Giacomo Marciani <gmarciani@acm.org>
 * @Institution: University of Rome Tor Vergata
 *
 * @Usage: matrix_transfer_2d matrixRows matrixCols matrixZ blockSize
 *
 * Default values:
 *  matrixRows: 4096
 *  matrixCols: 4096
 *  matrixZ:    4096
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

__global__ void matrixCopy(REAL *a, REAL *b, const unsigned int matrixRows, const unsigned int matrixCols, const unsigned int matrixZ) {
  const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int z   = blockIdx.z * blockDim.z + threadIdx.z;

  if (row >= matrixRows || col >= matrixCols || z >= matrixZ) return;

  const unsigned int pos = (z * matrixRows * matrixCols) + (row * matrixCols) + col;

  b[pos] = a[pos];
}

__host__ void gpuMatrixCopy(REAL ***a, REAL ***b, const unsigned int matrixRows, const unsigned int matrixCols, const unsigned int matrixZ, const dim3 gridDim, const dim3 blockDim) {
  REAL *dev_a = NULL; // device copies of a, b
  REAL *dev_b = NULL; // device copies of a, b
  const size_t size = matrixRows * matrixCols * matrixZ * sizeof(REAL); // bytes for a, b
  const size_t sizeX = matrixCols * sizeof(REAL); // bytes for a, b (dimension X)
  unsigned int z, r; // indices

  // allocate device copies of a, b
  HANDLE_ERROR(cudaMalloc((void**)&dev_a, size));
  HANDLE_ERROR(cudaMalloc((void**)&dev_b, size));

  // copy inputs to device
  for (z = 0; z < matrixZ; z++) {
    for (r = 0; r < matrixRows; r++) {
      HANDLE_ERROR(cudaMemcpy((void*)(dev_a + (z * matrixRows * matrixCols) + (r * matrixCols)), (const void*)a[z][r], sizeX, cudaMemcpyHostToDevice));
    }
  }

  // launch kernel matrixCopy()
  matrixCopy<<< gridDim, blockDim >>>(dev_a, dev_b, matrixRows, matrixCols, matrixZ);

  // copy device result back to host copy of b
  for (z = 0; z < matrixZ; z++) {
    for (r = 0; r < matrixRows; r++) {
      HANDLE_ERROR(cudaMemcpy((void*)b[z][r], (const void*)(dev_b + (z * matrixRows * matrixCols) + (r * matrixCols)), sizeX, cudaMemcpyDeviceToHost));
    }
  }

  // free device
  HANDLE_ERROR(cudaFree(dev_a));
  HANDLE_ERROR(cudaFree(dev_b));
}

int main(const int argc, const char **argv) {
  REAL ***a, ***b = NULL; // host copies of a, b
  unsigned int sizeX, sizeY, sizeZ; // bytes for a, b
  unsigned int matrixRows, matrixCols, matrixZ; // matrix dimensions
  unsigned int gridSizeX, gridSizeY, gridSizeZ; // grid size
  unsigned int blockSize; // block size
  cudaDeviceProp gpuInfo; // gpu properties
  unsigned int r, c, z; // indices

  // check arguments
  if (argc < 5) {
    fprintf(stderr, "Usage: %s matrixRows matrixCols matrixZ blockSize\n", argv[0]);
    exit(1);
  }

  matrixRows = atoi(argv[1]);
  matrixCols = atoi(argv[2]);
  matrixZ    = atoi(argv[3]);
  blockSize  = atoi(argv[4]);

  if (matrixRows < 1) {
    fprintf(stderr, "Error: matrixRows expected >= 1, got %d\n", matrixRows);
    exit(1);
  }

  if (matrixCols < 1) {
    fprintf(stderr, "Error: matrixCols expected >= 1, got %d\n", matrixCols);
    exit(1);
  }

  if (matrixZ < 1) {
    fprintf(stderr, "Error: matrixZ expected >= 1, got %d\n", matrixZ);
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
  gridSizeZ = matrixZ / blockSize;
  if (gridSizeZ * blockSize < matrixZ) {
     gridSizeZ += 1;
  }
  dim3 gridDim(gridSizeX, gridSizeY, gridSizeZ);
  dim3 blockDim(blockSize, blockSize, blockSize);

  sizeZ = matrixZ * sizeof(REAL**);
  sizeY = matrixRows * sizeof(REAL*);
  sizeX = matrixCols * sizeof(REAL);

  HANDLE_ERROR(cudaGetDeviceProperties(&gpuInfo, 0));

  printf("-----------------------------------------\n");
  printf("3D Matrix (NxMxZ) Floating-Point Transfer\n");
  printf("-----------------------------------------\n");
  #ifdef DOUBLE
  printf("FP Precision: Double\n");
  #else
  printf("FP Precision: Single\n");
  #endif
  printf("Matrix Dimension: (%d, %d, %d)\n", matrixRows, matrixCols, matrixZ);
  printf("Grid Size: (%d, %d, %d) (max: (%d, %d, %d))\n",
    gridDim.x, gridDim.y, gridDim.z,
    gpuInfo.maxGridSize[0], gpuInfo.maxGridSize[1], gpuInfo.maxGridSize[2]);
  printf("Block Size: (%d, %d, %d) (max: (%d, %d, %d))\n",
    blockDim.x, blockDim.y, blockDim.z,
    gpuInfo.maxThreadsDim[0], gpuInfo.maxThreadsDim[1], gpuInfo.maxThreadsDim[2]);
  printf("-----------------------------------\n");

  // allocate host copies of a, b
  HANDLE_NULL(a = (REAL***)malloc(sizeZ));
  for (z = 0; z < matrixZ; z++) {
    HANDLE_NULL(a[z] = (REAL**)malloc(sizeY));
    for (r = 0; r < matrixRows; r++) {
      HANDLE_NULL(a[z][r] = (REAL*)malloc(sizeX));
    }
  }

  HANDLE_NULL(b = (REAL***)malloc(sizeZ));
  for (z = 0; z < matrixZ; z++) {
    HANDLE_NULL(b[z] = (REAL**)malloc(sizeY));
    for (r = 0; r < matrixRows; r++) {
      HANDLE_NULL(b[z][r] = (REAL*)malloc(sizeX));
    }
  }

  // fill a with random data
  #ifdef DOUBLE
  random_matrix_double_3d(a, matrixRows, matrixCols, matrixZ);
  #else
  random_matrix_float_3d(a, matrixRows, matrixCols, matrixZ);
  #endif

  // launch kernel matrixCopy()
  gpuMatrixCopy(a, b, matrixRows, matrixCols, matrixZ, gridDim, blockDim);

  // test result
  bool err = false;
  for (z = 0; z < matrixZ && !err; z++) {
    for (r = 0; r < matrixRows && !err; r++) {
      for (c = 0; c < matrixCols && !err; c++) {
        if (a[z][r][c] != b[z][r][c]) {
          err = true;
          break;
        }
      }
    }
  }

  if (err) {
    fprintf(stderr, "Error\n");
  } else {
    printf("Correct!\n");
  }

  // free host
  for (z = 0; z < matrixZ; z++) {
    for (r = 0; r < matrixRows; r++) {
      free(a[z][r]);
    }
    free(a[z]);
  }
  free(a);

  for (z = 0; z < matrixZ; z++) {
    for (r = 0; r < matrixRows; r++) {
      free(b[z][r]);
    }
    free(b[z]);
  }
  free(b);

  return 0;
}
