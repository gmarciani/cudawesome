/*
 * @Name: vector_dot_blocks_threads.cu
 * @Description: Integer vectors dot-product.
 * Multiple blocks, multiple threads per block.
 *
 * @Author: Giacomo Marciani <gmarciani@acm.org>
 * @Institution: University of Rome Tor Vergata
 */

#include <stdio.h>
#include <math.h>
#include "../../common/error.h"
#include "../../common/random.h"
#include "../../common/vector.h"

#ifdef DOUBLE
#define REAL double
#else
#define REAL float
#endif

#define VECTOR_DIM 4096
#define BLOCK_SIZE 32
#define GRID_SIZE (VECTOR_DIM / BLOCK_SIZE)

/*
__device__ void device_vector_print_float(const unsigned int bid, const unsigned int tid, const char *name, const float *a, const unsigned int dim) {
  printf("[%d,%d] > %s=[\n", bid, tid, name);
  for (unsigned int i = 0; i < dim; i++) {
    printf("(block: %d | i:%d): %f \n", bid, i, a[i]);
  }
  printf("]\n");
}
*/

__global__ void dot(const REAL *a, const REAL *b, REAL *c) {
  __shared__ REAL temp[BLOCK_SIZE];

  const unsigned int tid = threadIdx.x;
  const unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;

  //if (0 == tid) device_vector_print_float(blockIdx.x, tid, "temp-initial", temp, BLOCK_SIZE);

  //__syncthreads();

  temp[tid] = a[pos] * b[pos];

  //__syncthreads();

  //if (0 == tid) device_vector_print_float(blockIdx.x, tid, "temp-product", temp, BLOCK_SIZE);

  __syncthreads();

  for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
    if (tid % (2 * stride) == 0) {
      temp[tid] += temp[tid + stride];
    }
    __syncthreads();
    //if (0 == tid) device_vector_print_float(blockIdx.x, tid, "temp_stride", temp, BLOCK_SIZE);
  }

  if (0 == tid) {
    c[blockIdx.x] = temp[0];
  }
}

int main(void) {
  REAL *a, *b, *c, result;     // host copies of a, b, c, result
  REAL *dev_a, *dev_b, *dev_c; // device copies of a, b, c
  const unsigned int size_a_b = VECTOR_DIM * sizeof(REAL); // bytes for a, b
  const unsigned int size_c = GRID_SIZE * sizeof(REAL); // bytes for c

  printf("----------------------------------------\n");
  printf("Vector Dot Product (reduction: baseline)\n");
  printf("----------------------------------------\n");
  #ifdef DOUBLE
  printf("Precision: double\n");
  #else
  printf("Precision: single\n");
  #endif
  printf("Vector Dimension: %d\n", VECTOR_DIM);
  printf("Grid Size: %d\n", GRID_SIZE);
  printf("Block Size: %d\n", BLOCK_SIZE);
  printf("----------------------------------------\n");

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
  random_vector_double(a, VECTOR_DIM);
  random_vector_double(b, VECTOR_DIM);
  #else
  random_vector_float(a, VECTOR_DIM);
  random_vector_float(b, VECTOR_DIM);
  #endif
  /*
  for (unsigned int i = 0; i < VECTOR_DIM; i++) {
    a[i] = i;
    b[i] = i;
  }
  */

  //vector_print_float("A", a, VECTOR_DIM);
  //vector_print_float("B", b, VECTOR_DIM);

  // copy inputs to device
  HANDLE_ERROR(cudaMemcpy(dev_a, a, size_a_b, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_b, b, size_a_b, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemset(dev_c, 0.0f, size_c));

  // launch dot() kernel
  dot<<< GRID_SIZE, BLOCK_SIZE >>>(dev_a, dev_b, dev_c);

  //cudaDeviceSynchronize();

  // copy device result back to host copy of c
  HANDLE_ERROR(cudaMemcpy(c, dev_c, size_c, cudaMemcpyDeviceToHost));

  //vector_print_float("C", c, GRID_SIZE);

  // reduce blocks result
  result = 0.0f;
  for (unsigned int block = 0; block < GRID_SIZE; block++) {
    result += c[block];
  }

  // test result
  REAL expected;
  #if DOUBLE
  vector_dot_double(a, b, &expected, VECTOR_DIM);
  #else
  vector_dot_float(a, b, &expected, VECTOR_DIM);
  #endif
  if (result != expected) {
    fprintf(stderr, "Error: expected %f, got %f\n", expected, result);
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
