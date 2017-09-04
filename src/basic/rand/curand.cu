/*
 * @Program: curand.cu
 * @Description: A simple cuRand example.
 *
 * @Author: Giacomo Marciani <gmarciani@acm.org>
 * @Institution: University of Rome Tor Vergata
 */

#include <stdlib.h>
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>

#ifdef DOUBLE
#define REAL double
#else
#define REAL float
#endif

#define DIM 10

#define N 1

__global__ void __initRandomGenerator(unsigned int seed, curandState_t* states) {

  /* we have to initialize the state */
  curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
              blockIdx.x, /* the sequence number should be different for each core (unless you want all
                             cores to get the same sequence of numbers for some reason - use thread id! */
              0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
              &states[blockIdx.x]);
}

__global__ void __randNumbers(double *dev_numbers, curandState_t* states) {
  unsigned int idx = blockDim.x * blockDim.x + threadIdx.x;

  dev_numbers[idx] = curand(&states[blockIdx.x]);
}

__host__ void gpuRand(REAL *a, const unsigned int dim, const dim3 gridDim, const dim3 blockDim) {
  curandState_t* states;
  REAL *dev_a;
  const unsigned int size = dim * sizeof(REAL); // bytes for dev_a

  // allocate device copy of a
  HANDLE_ERROR(cudaMalloc((void**) &dev_a, size));

  // allocate device copy of states
  HANDLE_ERROR(cudaMalloc((void**) &states, N * sizeof(curandState_t)));

  __initRandomGenerator<<<gridDim, 1>>>(time(0), states);

  __randNumbers<<< gridDim, blockDim >>>(dev_a, states);

  // copy device result back to host copy of a
  HANDLE_ERROR(cudaMemcpy(a, dev_a, size, cudaMemcpyDeviceToHost));

  // free device
  HANDLE_ERROR(cudaFree(dev_a));
  HANDLE_ERROR(cudaFree(states));

}

int main(void) {
  REAL *a;
  const unsigned int size = DIM * sizeof(REAL); // bytes for a

  // allocate host copy of a
  HANDLE_NULL(a = (REAL*)malloc(size));

  // grid settings
  dim3 gridDim(1);
  dim3 blockDim(1);

  // launch kernel gpuRand()
  gpuRand(a, DIM, gridDim, blockDim);

  int i;
  printf("[ ");
  for (i = 0; i < DIM; i++) {
    printf("%f ", a[i]);
  }
  printf("]");

  // free host
  free(a);

  return 0;
}
