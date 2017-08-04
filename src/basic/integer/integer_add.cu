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

int main(const int argc, char **argv) {
  int a, b, c; // host copies of a, b, c
  int *dev_c;  // device copy of c

  // check arguments
  if (argc < 3) {
    fprintf(stderr, "Usage: %s a b\n", argv[0]);
    exit(1);
  }

  // set values
  a = atoi(argv[1]);
  b = atoi(argv[2]);

  // allocate debice copy of c
  HANDLE_ERROR(cudaMalloc((void**)&dev_c, sizeof(int)));

  // lauch add() kernel
  add<<< 1, 1 >>>(a, b, dev_c);

  // copy device result back to host copy of c
  HANDLE_ERROR(cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost));

  // test result
  const int expected = a + b;
  if (c != expected) {
    fprintf(stderr, "Error: expected %d, got %d\n", expected, c);
  } else {
    printf("Correct: %d\n", c);
  }

  // free device
  HANDLE_ERROR(cudaFree(dev_c));

  return 0;
}
