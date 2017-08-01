/*
 * @Name: dotvec_blocks_threads.cu
 * @Description: Integer vectors dot-product with CUDA.
 * Multiple blocks, multiple threads per block.
 *
 * @Author: Giacomo Marciani <gmarciani@acm.org>
 */

#include <stdio.h>
#include <math.h>
#include <../common/error.h>

#define N 512
#define THREADS_PER_BLOCK 16

__global__ void dot(int *a, int *b, int *c) {
  __shared__ int temp[N];

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  temp[idx] = a[idx] * b[idx];

  __syncthreads();

  if (0 == threadIdx.x) {
    int sum = 0;
    int i;
    for(i = N - 1; i >=0; i--) {
      sum += temp[i];
    }
  }

  atomicAdd(c, sum);
}

void random_ints(int *p, int n) {
  int i;
  for(i = 0; i<n; i++) {
    p[i] = rand();
  }
}

int main( void ) {
  int *a, *b, *c;             // host copies of a, b, c
  int *dev_a, *dev_b, *dev_c; // device copies of a, b, c
  int size = N * sizeof(int); // size of N integers
  int i;

  // allocate device copies of a, b, c
  HANDLE_ERROR(cudaMalloc((void**)&dev_a, size));
  HANDLE_ERROR(cudaMalloc((void**)&dev_b, size));
  HANDLE_ERROR(cudaMalloc((void**)&dev_c, sizeof(int)));

  a = HANDLE_NULL((int*)malloc(size));
  b = HANDLE_NULL((int*)malloc(size));
  c = HANDLE_NULL((int*)malloc(sizeof(int)));

  random_ints(a, N);
  random_ints(b, N);

  // copy inputs to device
  HANDLE_ERROR(cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice));

  // launch add() kernel with N parallel blocks
  add<<< N / THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>(dev_a, dev_b, dev_c);

  // copy device result back to host copy of c
  HANDLE_ERROR(cudaMemcpy(c, dev_c, sizeof(int), cudaMemcpyDeviceToHost));

  int expected = 0;
  for(i = 0; i < N; i++) {
    expected += a[i] * b[i];
  }

  if(*c != expected) {
    printf("error: expected %d, got %d!\n", expected, *c);
    break;
  }

  free(a);
  free(b);
  free(c);

  HANDLE_ERROR(cudaFree(dev_a));
  HANDLE_ERROR(cudaFree(dev_b));
  HANDLE_ERROR(cudaFree(dev_c));

  return 0;
}
