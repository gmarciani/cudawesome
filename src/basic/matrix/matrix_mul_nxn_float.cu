/*
 * @Name: matrix_mul_nxn_float.cu
 * @Description: Matrix (NxN) Floating-Point Product.
 * Each matrix is viewed as a single block of memory.
 * Blocks and threads are viewed as a 2D grid.
 * Custom matrix dimension and block size.
 *
 * @Author: Giacomo Marciani <gmarciani@acm.org>
 * @Institution: University of Rome Tor Vergata
 *
 * @Usage: matrix_mul_nxn_float matrixDim blockSize
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

#ifdef DOUBLE
#define REAL double
#else
#define REAL float
#endif

#define EPSILON (float)1e-5

__global__ void matrixMul(const REAL *a, const REAL *b, REAL *c, const unsigned int dim) {
  const unsigned int iX = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int iY = blockIdx.y * blockDim.y + threadIdx.y;

  if (iX >= dim || iY >= dim) return;

  const unsigned int pos = iY * dim + iX;

  REAL val = 0.0f;
  for (unsigned int k = 0; k < dim; k++) {
    val += a[iY * dim + k] * b[k * dim + iX];
  }

  c[pos] = val;
}

__host__ void gpuMatrixMul(const REAL *a, const REAL *b, REAL *c, const unsigned int matrixDim, const dim3 gridDim, const dim3 blockDim) {
  REAL *dev_a, *dev_b, *dev_c; // device copies of a, b, c
  const unsigned int size = matrixDim * matrixDim * sizeof(REAL); // bytes for a, b, c

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
  REAL *a, *b, *c;         // host copies of a, b, c
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

  size = matrixDim * matrixDim * sizeof(REAL);

  HANDLE_ERROR(cudaGetDeviceProperties(&gpuInfo, 0));

  printf("------------------------------------\n");
  printf("Matrix (NxM) Floating-Point Product\n");
  printf("------------------------------------\n");
  #ifdef DOUBLE
  printf("FP Precision: Double\n");
  #else
  printf("FP Precision: Single\n");
  #endif
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
  HANDLE_NULL(a = (REAL*)malloc(size));
  HANDLE_NULL(b = (REAL*)malloc(size));
  HANDLE_NULL(c = (REAL*)malloc(size));

  // fill a, b with random data
  #ifdef DOUBLE
  random_matrix_double(a, matrixDim, matrixDim);
  random_matrix_double(b, matrixDim, matrixDim);
  #else
  random_matrix_float(a, matrixDim, matrixDim);
  random_matrix_float(b, matrixDim, matrixDim);
  #endif

  // launch kernel matrixMul()
  gpuMatrixMul(a, b, c, matrixDim, gridDim, blockDim);

  // test result
  REAL *expected;
  HANDLE_NULL(expected = (REAL*)malloc(size));
  #ifdef DOUBLE
  matrix_mul_double(a, b, expected, matrixDim, matrixDim, matrixDim);
  const bool correct = matrix_equals_err_double(c, expected, matrixDim, matrixDim, EPSILON);
  #else
  matrix_mul_float(a, b, expected, matrixDim, matrixDim, matrixDim);
  const bool correct = matrix_equals_err_float(c, expected, matrixDim, matrixDim, EPSILON);
  #endif
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
