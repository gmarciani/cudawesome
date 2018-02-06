/*
 * @Name: vector_dot_int_2.cu
 * @Description: Vector Floating-Point Dot Product.
 * Multiple blocks, multiple threads per block.
 *
 * @Author: Giacomo Marciani <gmarciani@acm.org>
 * @Institution: University of Rome Tor Vergata
 *
 * @Usage: vector_dot_int_2 vectorDim blockSize
 *
 * Default values:
 *  vectorDim: 1048576
 *  blockSize: 256
 *
 * @See: http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf
 */

#include <stdio.h>
#include <math.h>
#include "../../../common/error.h"
#include "../../../common/random.h"
#include "../../../common/vector.h"
#include "../../../common/mathutil.h"

#ifdef DOUBLE
#define REAL double
#else
#define REAL float
#endif

#define EPSILON (float)1e-5

__global__ void vectorDot(const REAL *a, const REAL *b, REAL *c, const unsigned int vectorDim) {
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

__host__ void gpuVectorDot(const REAL *a, const REAL *b, REAL *result, const unsigned int vectorDim, const dim3 gridDim, const dim3 blockDim) {
  REAL *dev_a, *dev_b, *dev_partial; // device copies of a, b, partial
  REAL *partial; // host copy for partial result
  const unsigned int size_a_b = vectorDim * sizeof(REAL); // bytes for a, b
  const unsigned int size_partial = gridDim.x * sizeof(REAL); // bytes for partial

  // allocate host copies of partial
  HANDLE_NULL(partial = (REAL*)malloc(size_partial));

  // allocate device copies of a, b, c
  HANDLE_ERROR(cudaMalloc((void**)&dev_a, size_a_b));
  HANDLE_ERROR(cudaMalloc((void**)&dev_b, size_a_b));
  HANDLE_ERROR(cudaMalloc((void**)&dev_partial, size_partial));

  // copy inputs to device
  HANDLE_ERROR(cudaMemcpyAsync(dev_a, a, size_a_b, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpyAsync(dev_b, b, size_a_b, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemsetAsync(dev_partial, 0.0f, size_partial));

  // shared memory settings
  const unsigned int sharedMemSize = (unsigned int) blockDim.x * sizeof(REAL);

  // launch kernel vectorDot
  vectorDot<<< gridDim, blockDim, sharedMemSize >>>(dev_a, dev_b, dev_partial, vectorDim);

  // copy device result back to host copy of c
  HANDLE_ERROR(cudaMemcpy(partial, dev_partial, size_partial, cudaMemcpyDeviceToHost));

  // reduce blocks result
  *result = 0.0f;
  for (unsigned int block = 0; block < gridDim.x; block++) {
    (*result) += partial[block];
  }

  // free host
  free(partial);

  // free device
  HANDLE_ERROR(cudaFree(dev_a));
  HANDLE_ERROR(cudaFree(dev_b));
  HANDLE_ERROR(cudaFree(dev_partial));
}

int main(const int argc, const char **argv) {
  REAL *a, *b, result; // host copies of a, b, result
  unsigned int vectorDim; // vector dimension
  unsigned int size_a_b; // bytes for a, b
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

  HANDLE_ERROR(cudaGetDeviceProperties(&gpuInfo, 0));

  printf("----------------------------------\n");
  printf("Vector Floating-Point Dot Product\n");
  printf("Reduction: sequential addressing\n");
  printf("----------------------------------\n");
  #ifdef DOUBLE
  printf("FP Precision: Double\n");
  #else
  printf("FP Precision: Single\n");
  #endif
  printf("Vector Dimension: %d\n", vectorDim);
  printf("Grid Size: (%d %d %d) (max: (%d %d %d))\n",
    gridDim.x, gridDim.y, gridDim.z,
    gpuInfo.maxGridSize[0], gpuInfo.maxGridSize[1], gpuInfo.maxGridSize[2]);
  printf("Block Size: (%d %d %d) (max: (%d %d %d))\n",
    blockDim.x, blockDim.y, blockDim.z,
    gpuInfo.maxThreadsDim[0], gpuInfo.maxThreadsDim[1], gpuInfo.maxThreadsDim[2]);
  printf("---------------------------------\n");

  // allocate host copies of a, b, c
  HANDLE_NULL(a = (REAL*)malloc(size_a_b));
  HANDLE_NULL(b = (REAL*)malloc(size_a_b));

  // fill a, b with random data
  #ifdef DOUBLE
  random_vector_double(a, vectorDim);
  random_vector_double(b, vectorDim);
  #else
  random_vector_float(a, vectorDim);
  random_vector_float(b, vectorDim);
  #endif

  // launch kernel vectorDot()
  gpuVectorDot(a, b, &result, vectorDim, gridDim, blockDim);

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

  return 0;
}
