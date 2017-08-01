/*
 * @Name: addint.cu
 * @Description: Integer addition.
 * Arguments passed as values.
 *
 * @Author: Giacomo Marciani <gmarciani@acm.org>
 * @Institution: University of Rome Tor Vergata
 */

#include <stdio.h>
#include "../common/error.h"

__global__ void add(int a, int b, int *c) {
  *c = a + b;
}

int main(void) {
  int c;      // host copy of c
  int *dev_c; // device copy of c

  // allocate debice copy of c
  HANDLE_ERROR(cudaMalloc((void**)&dev_c, sizeof(int)));

  // lauch add() kernel
  add<<< 1, 1 >>>(2, 7, dev_c);

  // copy device result back to host copy of c
  HANDLE_ERROR(cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost));

  // print result
  printf("2 + 7 = %d\n", c);

  // free device
  HANDLE_ERROR(cudaFree(dev_c));

  return 0;
}
