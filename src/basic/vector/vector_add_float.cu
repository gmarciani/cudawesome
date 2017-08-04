/*
 * @Name: vector_add_float.cu
 * @Description: Vector Floating-Point Sum
 * Custom vector dimension and block size.
 *
 * @Author: Giacomo Marciani <gmarciani@acm.org>
 * @Institution: University of Rome Tor Vergata
 *
 * @Usage: vector_add_float vectorDimension blockSize
 *
 * Default values:
 *  vectorDimension: 4096
 *  blockSize: 32
 */

#include <stdio.h>
#include <math.h>
#include "../../common/error.h"
#include "../../common/random.h"
#include "../../common/vector.h"
#include "../../common/mathutil.h"

#ifdef DOUBLE
#define REAL double
#else
#define REAL float
#endif

#define EPSILON (float)1e-5

__global__ void add(const REAL *a, const REAL *b, REAL *c, const unsigned int dim) {
  const unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;

  if (pos >= dim) return;

  c[pos] = a[pos] + b[pos];
}

__host__ gpuAdd(const REAL *a, const REAL *b, REAL *c, const dim3 grid, const dim3 block) {

}

int main(const int argc, const char **argv) {
  REAL *a, *b, *c;             // host copies of a, b, c
  REAL *dev_a, *dev_b, *dev_c; // device copies of a, b, c
  unsigned int size; // bytes for a, b, c
  unsigned int vectorDim; // vector dimension
  unsigned int gridSize;  // grid size
  unsigned int blockSize; // block size
  cudaDeviceProp gpuInfo; // gpu properties

  // check arguments
  if (argc < 3) {
    fprintf(stderr, "Usage: %s vectorDim blockSize\n", argv[0]);
    exit(1);
  }

  vectorDim = atoi(argv[1]);
  blockSize = atoi(argv[2]);

  if (vectorDim < 1) {
    fprintf(stderr, "Error: vectorDim expected >= 1, got %d\n", vectorDim);
    exit(1);
  }

  if (!IS_POWER_OF_2(blockSize)) {
    fprintf(stderr, "Error: blockSize expected as power of 2, got %d\n", blockSize);
    exit(1);
  }

  // grid settings
  gridSize = vectorDim / blockSize;
  if (gridSize * blockSize < vectorDim) {
    gridSize += 1;
  }
  dim3 gridDim(gridSize);
  dim3 blockDim(blockSize);

  size = vectorDim * sizeof(REAL);

  HANDLE_ERROR(cudaGetDeviceProperties(&gpuInfo, 0));

  printf("----------------------------------\n");
  printf("Vector Floating-Point Sum\n");
  printf("----------------------------------\n");
  #ifdef DOUBLE
  printf("FP Precision: Double\n");
  #else
  printf("FP Precision: Single\n");
  #endif
  printf("Vector Dimension: %d\n", vectorDim);
  printf("Grid Size: %d (max: %d)\n", gridSize, gpuInfo.maxGridSize[0]);
  printf("Block Size: %d (max: %d)\n", blockSize, gpuInfo.maxThreadsDim[1]);
  printf("---------------------------------\n");

  // allocate host copies of a, b, c
  HANDLE_NULL(a = (REAL*)malloc(size));
  HANDLE_NULL(b = (REAL*)malloc(size));
  HANDLE_NULL(c = (REAL*)malloc(size));

  // allocate device copies of a, b, c
  HANDLE_ERROR(cudaMalloc((void**)&dev_a, size));
  HANDLE_ERROR(cudaMalloc((void**)&dev_b, size));
  HANDLE_ERROR(cudaMalloc((void**)&dev_c, size));

  // fill a, b with random data
  #ifdef DOUBLE
  random_vector_double(a, vectorDim);
  random_vector_double(b, vectorDim);
  #else
  random_vector_float(a, vectorDim);
  random_vector_float(b, vectorDim);
  #endif

  // copy inputs to device
  HANDLE_ERROR(cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice));

  // launch add() kernel
  add<<< gridDim, blockDim >>>(dev_a, dev_b, dev_c, vectorDim);

  // copy device result back to host copy of c
  HANDLE_ERROR(cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost));

  // test result
  REAL *d;
  HANDLE_NULL(d = (REAL*)malloc(size));
  #ifdef DOUBLE
  vector_add_double(a, b, d, vectorDim);
  const bool equal = vector_equals_err_double(c, d, vectorDim, EPSILON);
  #else
  vector_add_float(a, b, d, vectorDim);
  const bool equal = vector_equals_err_float(c, d, vectorDim, EPSILON);
  #endif
  if (!equal) {
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
