/*
 * @Name: mtxset_nxn_grid2.cu
 * @Description: Sets the elements of an integer NxM matrix.
 * The matrix is viewed as a single block of memory.
 * Blocks and threads are viewed as a 2D grid.
 *
 * @Author: Giacomo Marciani <gmarciani@acm.org>
 * @Institution: University of Rome Tor Vergata
 */

#include <stdio.h>
#include <math.h>
#include <../common/error.h>

#define MATRIX_DIM_X 9
#define MATRIX_DIM_Y 6

__global__ void set(int *a, int dim) {
  int iX = blockIdx.x * blockDim.x + threadIdx.x;
  int iY = blockIdx.y * blockDim.y + threadIdx.y;
  int idx = iY * dim + iX;

  a[idx] = idx;
}

int main( void ) {
  int *a;     // host copy of a
  int *dev_a; // device copy of a
  int size = MATRIX_DIM_X * MATRIX_DIM_Y * sizeof(int); // bytes for a matrix of MATRIX_DIM_X x MATRIX_DIM_Y integers

  // allocate host copy of a
  a = HANDLE_NULL((int*)malloc(size));

  // allocate device copy of a
  HANDLE_ERROR(cudaMalloc((void**)&dev_a, size));

  dim3 gridDim, blockDim;
  blockDim.x = 3;
  blockDim.y = 3;
  gridDim.x = MATRIX_DIM_X / blockDim.x;
  gridDim.y = MATRIX_DIM_Y / blockDim.y;

  // launch set() kernel
  set<<< gridDim, blockDim >>>(dev_a, MATRIX_DIM);

  // copy device result back to host copy of a
  HANDLE_ERROR(cudaMemcpy(a, dev_a, size, cudaMemcpyDeviceToHost));

  // print result
  for(int y = 0; y < MATRIX_DIM_Y; y++) {
    for(int x = 0; x < MATRIX_DIM_X; x++) {
        printf("%d ", a[y * MATRIX_DIM_X + x])
    }
    printf("\n");
  }

  // free host
  free(a);

  // free device
  HANDLE_ERROR(cudaFree(dev_a));

  return 0;
}
