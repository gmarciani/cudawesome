/*
 * @Name: ps.cu
 * @Description: CUDA kernels and wrappers for Prefix-Sum.
 *
 * @Author: Giacomo Marciani <gmarciani@acm.org>
 *          Gabriele Santi   <gsanti@acm.org>
 *
 * @Institution: University of Rome Tor Vergata
 */

#include <stdio.h>
#include <cuda_runtime.h>

#include "common/error.h"

#include "ps.cuh"
#include "psutil.cuh"


/*------------------------------------------------------------------------------
  WRAPPERS
*******************************************************************************/

/*------------------------------------------------------------------------------
  @description   Execute the Prefix-Sum operation (Host Wrapper).
  @param   input      The input data.
  @param   output     The output data.
  @param   dim_data  The number of elements in data.
  @return  void.
  ----------------------------------------------------------------------------*/
__host__ void prefix_sum(int *input, int *output, int dim_data) {

  /*----------------------------------------------------------------------------
    MAIN VARIABLES
  *****************************************************************************/
  // Input/Output data (Device)
  int *dev_data = NULL;

  // Vector of partial sums (Device)
  int *dev_partialSums = NULL;

  /*----------------------------------------------------------------------------
    RESOURCES ALLOCATION AND INITIALIZATION
  *****************************************************************************/
  /*---------------------------------------------------------------------------+
  |  Device Allocations and Initializations:
  |     * dev_data: input data, array of integers.
  |     * dev_partialSums: partial sums by blocks, array of integers.
  +---------------------------------------------------------------------------*/
  // Dimensions
  const int dim_partialSums  = dim_data / BLOCK_SIZE; // number of elements in partialSums

  // Sizes
  const size_t size_data = sizeof(int) * dim_data; // bytes in data
  const size_t size_partialSums = sizeof(int) * dim_partialSums; // bytes in partialSums

  // Allocations
  HANDLE_ERROR(cudaMalloc((void **)&dev_data, size_data));
  HANDLE_ERROR(cudaMalloc((void **)&dev_partialSums, size_partialSums));

  // Initializations
  HANDLE_ERROR(cudaMemcpy(dev_data, input, size_data, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemset(dev_partialSums, 0, size_partialSums));

  /*----------------------------------------------------------------------------
  * GRID SETTINGS
  *****************************************************************************/
  // Grid
  const dim3 grid(dim_data / BLOCK_SIZE, 1, 1);
  const dim3 block(BLOCK_SIZE, 1, 1);

  // Shared memory: shmem[i] contains the sum for warp i
  const int dim_sharedMemory = BLOCK_SIZE / WARP_SIZE; // number of elements in shared memory
  const size_t size_sharedMemory = sizeof(int) * dim_sharedMemory; // bytes in partialSums

  /*----------------------------------------------------------------------------
  * KERNEL LAUNCH
  *****************************************************************************/
  // If kernel profiling is active, register events for elapsed time calculation
  #ifdef PROFILE_KERNEL
  cudaEvent_t start, stop;
  float elapsed = 0;
  HANDLE_ERROR(cudaEventCreate(&start));
  HANDLE_ERROR(cudaEventCreate(&stop));
  HANDLE_ERROR(cudaEventRecord(start));
  #endif

  // Launch kernel for Prefix-Sum calculation
  __prefix_sum<<< grid, block, size_sharedMemory >>>(dev_data, dev_partialSums, WARP_SIZE);

  // If kernel profiling is active, register events for elapsed time calculation
  #ifdef PROFILE_KERNEL
  HANDLE_ERROR(cudaEventRecord(stop));
  HANDLE_ERROR(cudaEventSynchronize(stop));
  HANDLE_ERROR(cudaEventElapsedTime(&elapsed, start, stop));
  printf("Elapsed Time (ms): %f\n", elapsed);
  printf("%d elements scanned in %f ms -> %f MegaElements/s\n",
           dim_data, elapsed, dim_data/(elapsed/1000.0f)/1000000.0f);
  #endif

  HANDLE_ERROR(cudaMemcpy(output, dev_data, size_data, cudaMemcpyDeviceToHost));

  /*----------------------------------------------------------------------------
  * FREE RESOURCES (DEVICE)
  *****************************************************************************/
  HANDLE_ERROR(cudaFree(dev_data));
  HANDLE_ERROR(cudaFree(dev_partialSums));
}


/*------------------------------------------------------------------------------
 * KERNELS
 ******************************************************************************/
/*------------------------------------------------------------------------------
@description  Computes the Prefix-Sum on the input array.
@param   data         Input/Outut data (output is overwritten).
@param   partialSums  Partial sums, shared between blocks.
@param   warpSize     The warp size.
@return  void.
----------------------------------------------------------------------------*/
__global__ void __prefix_sum(int *data, int *partialSums, int warpSize) {
  // Shared memory: shmem[i] contains sum for warp i
  extern __shared__ int shmem[];

  // Declaration of: CellId, ThreadId and WarpId.
  const int id = (blockIdx.x * blockDim.x) + threadIdx.x;
  const int tid = threadIdx.x;
  const int warpId = tid / warpSize;

  int d, i; // Indices for loops

  // Initialize value with the input at cell i
  int value = data[id];

  /*----------------------------------------------------------------------------
    STEP 1 - WARP REDUCE
  *****************************************************************************/
  for ( d = 1; d < warpSize; d *= 2 ) {
    int temp = __shfl_up_sync(0xFFFFFFFF, value, d, warpSize); // CUDA 9
    if ( tid % warpSize >= d ) {
      value += temp;
    }
  }
  // The last thread whitin a warp writes its value on shared memory
  if ( tid % warpSize == (warpSize-1) ) {
    shmem[warpId] = value;
  }

  __syncthreads();

  /*----------------------------------------------------------------------------
    STEP 2 - BLOCK REDUCE
  *****************************************************************************/
  // Each warp (not the first) of a block updates value with sums of
  // previous warps
  if ( warpId > 0 ) {
    for ( i = 0; i < warpId; i++ ) {
      value += shmem[i];
    }
  }
  // The last thread of each block stores the partial sum of its block into
  // the vector of partial sums
  if ( threadIdx.x == (blockDim.x-1) ) {
    partialSums[blockIdx.x] = value;
  }

  __syncthreads();

  /*----------------------------------------------------------------------------
    STEP 3 - FINAL REDUCE
  *****************************************************************************/
  // Each thread of each block (not the first block) updates value with sums of
  // previous blocks
  if ( blockIdx.x > 0 ) {
    for ( i = 0; i < blockIdx.x; i++ ) {
      value += partialSums[i];
    }
  }
  // Update cell i with value
  data[id] = value;
}
