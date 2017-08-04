/*
 * @Name: integer_add_ptr.cu
 * @Description: Integer addition.
 * Arguments passed as pointers.
 * One block, one thread.
 *
 * @Author: Giacomo Marciani <gmarciani@acm.org>
 * @Institution: University of Rome Tor Vergata
 *
 * @Usage: add_int_ptr a b
 */

#include <stdio.h>
#include <math.h>
#include "../../common/error.h"

__global__ void add(const int *a, const int *b, int *c) {
  *c = *a + *b;
}

int main(const int argc, char **argv) {
  int a, b, c;                // host copies of a, b, c
  int *dev_a, *dev_b, *dev_c; // device copies of a, b, c
  const unsigned int size = sizeof(int); // bytes for and integer

  // check arguments
  if (argc < 3) {
    fprintf(stderr, "Usage: %s a b\n", argv[0]);
    exit(1);
  }

  // set values
  a = atoi(argv[1]);
  b = atoi(argv[2]);

  // allocate device copies of a, b, c
  HANDLE_ERROR(cudaMalloc((void**)&dev_a, size));
  HANDLE_ERROR(cudaMalloc((void**)&dev_b, size));
  HANDLE_ERROR(cudaMalloc((void**)&dev_c, size));

  // copy inputs to device
  HANDLE_ERROR(cudaMemcpy(dev_a, &a, size, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_b, &b, size, cudaMemcpyHostToDevice));

  // launch add() kernel
  add<<< 1, 1 >>>(dev_a, dev_b, dev_c);

  // copy device result back to host copy of c
  HANDLE_ERROR(cudaMemcpy(&c, dev_c, size, cudaMemcpyDeviceToHost));

  // test result
  const int expected = a + b;
  if (c != expected) {
    fprintf(stderr, "Error: expected %d, got %d\n", expected, c);
  } else {
    printf("Correct: %d\n", c);
  }

  // free device
  HANDLE_ERROR(cudaFree(dev_a));
  HANDLE_ERROR(cudaFree(dev_b));
  HANDLE_ERROR(cudaFree(dev_c));

  return 0;
}
