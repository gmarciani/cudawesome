/*
 * @Name: integer_add.cu
 * @Description: Integer addition.
 * Arguments passed as values.
 *
 * @Author: Giacomo Marciani <gmarciani@acm.org>
 * @Institution: University of Rome Tor Vergata
 *
 * @Usage: add_int a b
 */

#include <stdio.h>
#include "../../common/error.h"

__global__ void add(const int a, const int b, int *c) {
  *c = a + b;
}

__host__ void gpuAdd(const int a, const int b, int *c, const dim3 gridDim, const dim3 blockDim) {
  int *dev_c; // device copies of c
  const unsigned int size = sizeof(int); // bytes for and integer

  // allocate device copies of c
  HANDLE_ERROR(cudaMalloc((void**)&dev_c, size));

  // launch add() kernel
  add<<< gridDim, blockDim >>>(a, b, dev_c);

  // copy device result back to host copy of c
  HANDLE_ERROR(cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost));

  // free device
  HANDLE_ERROR(cudaFree(dev_c));
}

int main(const int argc, char **argv) {
  int a, b, c; // host copies of a, b, c

  // check arguments
  if (argc < 3) {
    fprintf(stderr, "Usage: %s a b\n", argv[0]);
    exit(1);
  }

  // set values
  a = atoi(argv[1]);
  b = atoi(argv[2]);

  // launch add() kernel
  dim3 gridDim(1);
  dim3 blockDim(1);
  gpuAdd(a, b, &c, gridDim, blockDim);

  // test result
  const int expected = a + b;
  if (c != expected) {
    fprintf(stderr, "Error: expected %d, got %d\n", expected, c);
  } else {
    printf("Correct: %d\n", c);
  }

  return 0;
}
