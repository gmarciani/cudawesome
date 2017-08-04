/*
 * @Name: vector_dot.cu
 * @Description: Integer vectors dot-product.
 * Custom vector dimension and block size.
 *
 * @Author: Giacomo Marciani <gmarciani@acm.org>
 * @Institution: University of Rome Tor Vergata
 *
 * @Usage: vector_dot vectorDimension blockSize
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

__global__ void dot(const REAL *a, const REAL *b, REAL *c, const unsigned int dim) {
  extern __shared__ REAL temp[];

  const unsigned int tid = threadIdx.x;
  const unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;

  if (pos < dim) {
    temp[tid] = a[pos] * b[pos];

    __syncthreads();

    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
      if (tid % (2 * stride) == 0) {
        temp[tid] += temp[tid + stride];
      }
      __syncthreads();
    }

    if (0 == tid) {
      c[blockIdx.x] = temp[0];
    }
  }
}

int main(const int argc, const char **argv) {
  REAL *a, *b, *c, result;     // host copies of a, b, c, result
  REAL *dev_a, *dev_b, *dev_c; // device copies of a, b, c
  unsigned int size_a_b; // bytes for a, b
  unsigned int size_c;   // bytes for c
  unsigned int vectorDim; // vector dimension
  unsigned int gridSize;  // grid size
  unsigned int blockSize; // block size

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

  #ifdef DOUBLE
  printf("Double precision\n");
  #else
  printf("Single precision\n");
  #endif

  size_a_b = vectorDim * sizeof(REAL);
  size_c = blockSize * sizeof(REAL);

  // allocate host copies of a, b, c
  HANDLE_NULL(a = (REAL*)malloc(size_a_b));
  HANDLE_NULL(b = (REAL*)malloc(size_a_b));
  HANDLE_NULL(c = (REAL*)malloc(size_c));

  // allocate device copies of a, b, c
  HANDLE_ERROR(cudaMalloc((void**)&dev_a, size_a_b));
  HANDLE_ERROR(cudaMalloc((void**)&dev_b, size_a_b));
  HANDLE_ERROR(cudaMalloc((void**)&dev_c, size_c));

  // fill a, b with random data
  #ifdef DOUBLE
  random_vector_double(a, vectorDim);
  random_vector_double(b, vectorDim);
  #else
  random_vector_float(a, vectorDim);
  random_vector_float(b, vectorDim);
  #endif

  // copy inputs to device
  HANDLE_ERROR(cudaMemcpy(dev_a, a, size_a_b, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_b, b, size_a_b, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemset(dev_c, 0.0f, size_c));

  // grid settings
  gridSize = vectorDim / blockSize;
  if (gridSize * blockSize < vectorDim) {
    gridSize += 1;
  }

  // shared memory settings
  const unsigned int sharedMemSize = (unsigned int) blockSize * sizeof(REAL);

  // launch dot() kernel
  dot<<< gridSize, blockSize, sharedMemSize >>>(dev_a, dev_b, dev_c, vectorDim);

  // copy device result back to host copy of c
  HANDLE_ERROR(cudaMemcpy(c, dev_c, size_c, cudaMemcpyDeviceToHost));

  // reduce blocks result
  result = 0.0f;
  for (unsigned int block = 0; block < blockSize; block++) {
    result += c[block];
  }

  // test result
  REAL expected;
  #if DOUBLE
  vector_dot_double(a, b, &expected, vectorDim);
  #else
  vector_dot_float(a, b, &expected, vectorDim);
  #endif
  if (result != expected) {
    fprintf(stderr, "Error\n");
  } else {
    printf("Correct\n");
  }

  // free host
  free(a);
  free(b);
  free(c);

  // free device
  HANDLE_ERROR(cudaFree(dev_a));
  HANDLE_ERROR(cudaFree(dev_b));
  HANDLE_ERROR(cudaFree(dev_c));

  return 0;
}
