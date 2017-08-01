/*******************************************************************************
* @Program: hello_world.cu
* @Description: The classic Hello World.
*
* @Author: Giacomo Marciani <gmarciani@acm.org>
* @Institution: University of Rome Tor Vergata
*******************************************************************************/

#include <stdlib.h>
#include <stdio.h>

__global__ void kernel(void) {

}

int main(void) {
  kernel<<< 1, 1 >>>();
  printf("Hello world!\n");
  return 0;
}
