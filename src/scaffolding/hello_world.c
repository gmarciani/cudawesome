/*
 * @Program: hello_world.cu
 * @Description: The classic Hello World.
 *
 * @Author: Giacomo Marciani <gmarciani@acm.org>
 * @Institution: University of Rome Tor Vergata
 */

#include <stdlib.h>
#include <stdio.h>

#include "include_c/cpu_functions.h"
#include "include_cu/gpu_functions.cuh"

int main(void) {

  helloGPU();

  helloCPU();

  return 0;
}
