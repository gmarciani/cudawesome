/*
 * @Program: hello_world.cu
 * @Description: The classic Hello World.
 *
 * @Author: Giacomo Marciani <gmarciani@acm.org>
 * @Institution: University of Rome Tor Vergata
 */

#include <stdlib.h>
#include <stdio.h>

__global__ void foo(void) {

}

int main(void) {
  // launch foo() kernel
  foo<<< 1, 1 >>>();

  // print result
  printf("Hello world!\n");

  return 0;
}
