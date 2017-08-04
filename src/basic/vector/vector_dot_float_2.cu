/*
 * @Name: vector_dot_int_2.cu
 * @Description: Vector Floating-Point Dot Product.
 * Multiple blocks, multiple threads per block.
 *
 * @Author: Giacomo Marciani <gmarciani@acm.org>
 * @Institution: University of Rome Tor Vergata
 *
 * @Usage: vector_dot_int_2 vectorDimension blockSize
 *
 * Default values:
 *  vectorDimension: 4096
 *  blockSize: 32
 *
 * @See: http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf
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

__global__ void dot(const REAL *a, const REAL *b, REAL *c, const unsigned int vectorDim) {
  extern __shared__ REAL temp[];

  const unsigned int tid = threadIdx.x;
  const unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;

  if (pos >= vectorDim) return;

  temp[tid] = a[pos] * b[pos];

  __syncthreads();

  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      temp[tid] += temp[tid + stride];
    }
    __syncthreads();
  }

  if (0 == tid) {
    c[blockIdx.x] = temp[0];
  }
}

int main(const int argc, const char **argv) {
  REAL *a, *b, *c, result;     // host copies of a, b, c, result
  REAL *dev_a, *dev_b, *dev_c; // device copies of a, b, c
  unsigned int size_a_b; // bytes for a, b
  unsigned int size_c; // bytes for c
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

  size_a_b = vectorDim * sizeof(REAL);
  size_c = gridSize * sizeof(REAL);

  HANDLE_ERROR(cudaGetDeviceProperties(&gpuInfo, 0));

  printf("---------------------------------\n");
  printf("Vector Floating-Point Dot Product\n");
  printf("Reduction: sequential addressing\n");
  printf("---------------------------------\n");
  #ifdef DOUBLE
  printf("FP Precision: Double\n");
  #else
  printf("FP Precision: Single\n");
  #endif
  printf("Vector Dimension: %d\n", vectorDim);
  printf("Grid Size: %d (max: %d)\n", gridSize, gpuInfo.maxGridSize[0]);
  printf("Block Size: %d (max: %d)\n", blockSize, gpuInfo.maxThreadsDim[1]);
  printf("--------------------------------\n");

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

  // shared memory settings
  const unsigned int sharedMemSize = (unsigned int) gridSize * sizeof(REAL);

  // launch dot() kernel
  dot<<< gridDim, blockDim, sharedMemSize >>>(dev_a, dev_b, dev_c, vectorDim);

  // copy device result back to host copy of c
  HANDLE_ERROR(cudaMemcpy(c, dev_c, size_c, cudaMemcpyDeviceToHost));

  // reduce blocks result
  result = 0.0f;
  for (unsigned int block = 0; block < gridSize; block++) {
    result += c[block];
  }

  // test result
  REAL expected;
  #if DOUBLE
  vector_dot_double(a, b, &expected, vectorDim);
  #else
  vector_dot_float(a, b, &expected, vectorDim);
  #endif
  if (fabs(expected - result) > EPSILON * expected) {
    fprintf(stderr, "Error: expected %f, got %f (error:%f %%)\n",
      expected, result, (fabs(expected - result) / expected) * 100.0);
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
