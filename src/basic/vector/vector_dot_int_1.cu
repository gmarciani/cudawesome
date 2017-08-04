/*
 * @Name: vector_dot_int_1.cu
 * @Description: Vector Integer Dot Product
 * Multiple blocks, multiple threads per block.
 *
 * @Author: Giacomo Marciani <gmarciani@acm.org>
 * @Institution: University of Rome Tor Vergata
 *
 * @Usage: vector_dot_int_1 vectorDimension blockSize
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

__global__ void dot(const int *a, const int *b, int *c, const unsigned int vectorDim) {
  extern __shared__ int temp[];

  const unsigned int tid = threadIdx.x;
  const unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;

  if (pos < vectorDim) {
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
  int *a, *b, *c, result;     // host copies of a, b, c, result
  int *dev_a, *dev_b, *dev_c; // device copies of a, b, c
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

  size_a_b = vectorDim * sizeof(int);
  size_c = gridSize * sizeof(int);

  HANDLE_ERROR(cudaGetDeviceProperties(&gpuInfo, 0));

  printf("----------------------------------\n");
  printf("Vector Integer Dot Product\n");
  printf("Reduction: interleaving addressing\n");
  printf("----------------------------------\n");
  printf("Vector Dimension: %d\n", vectorDim);
  printf("Grid Size: %d (max: %d)\n", gridSize, gpuInfo.maxGridSize[0]);
  printf("Block Size: %d (max: %d)\n", blockSize, gpuInfo.maxThreadsDim[1]);
  printf("----------------------------------\n");

  // allocate host copies of a, b, c
  HANDLE_NULL(a = (int*)malloc(size_a_b));
  HANDLE_NULL(b = (int*)malloc(size_a_b));
  HANDLE_NULL(c = (int*)malloc(size_c));

  // allocate device copies of a, b, c
  HANDLE_ERROR(cudaMalloc((void**)&dev_a, size_a_b));
  HANDLE_ERROR(cudaMalloc((void**)&dev_b, size_a_b));
  HANDLE_ERROR(cudaMalloc((void**)&dev_c, size_c));

  // fill a, b with random data
  random_vector_int(a, vectorDim);
  random_vector_int(b, vectorDim);

  // copy inputs to device
  HANDLE_ERROR(cudaMemcpy(dev_a, a, size_a_b, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_b, b, size_a_b, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemset(dev_c, 0, size_c));

  // shared memory settings
  const unsigned int sharedMemSize = (unsigned int) gridSize * sizeof(int);

  // launch dot() kernel
  dot<<< gridDim, blockDim, sharedMemSize >>>(dev_a, dev_b, dev_c, vectorDim);

  // copy device result back to host copy of c
  HANDLE_ERROR(cudaMemcpy(c, dev_c, size_c, cudaMemcpyDeviceToHost));

  // reduce blocks result
  result = 0;
  for (unsigned int block = 0; block < gridSize; block++) {
    result += c[block];
  }

  // test result
  int expected;
  vector_dot_int(a, b, &expected, vectorDim);
  if (result != expected) {
    fprintf(stderr, "Error: expected %d, got %d (error:%f %%)\n",
      expected, result, (((float)result - (float)expected) / (float)expected) * 100.0);
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
