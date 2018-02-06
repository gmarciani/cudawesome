/*
 * @Name: ps_cuda.cu
 * @Description: Prefix-Sum operation, leveraging CUDA.
 *
 * @Author: Giacomo Marciani <gmarciani@acm.org>
 *          Gabriele Santi   <gsanti@acm.org>
 *
 * @Institution: University of Rome Tor Vergata
 *
 * @Credits: Massimo Bernaschi
 *
 * @Usage: ps_cuda [numberOfElements]
 */

#include <stdio.h>
#include <cuda_runtime.h>

#include "common/mainutil.h"
#include "common/error.h"

#include "gpu/psutil.cuh"
#include "gpu/ps.cuh"


#define MAX_NUM_ELEMENTS 65536


/*------------------------------------------------------------------------------
  @description   The program entry-point.
  @param   argc         Number of arguments.
  @param   argv         Array of arguments.
  @return  isCorrect  The program result.
                        EXIT_SUCCESS if suceeded; EXIT_FAILURE, otherwise.
  ----------------------------------------------------------------------------*/
int main(int argc, char **argv) {

  /*----------------------------------------------------------------------------
  * VARIABLES
  *****************************************************************************/

  // Input data (Host)
  int *input = NULL;

  // Output data (Host)
  int *output = NULL;

  // Number of input elements
  int dim_data;


  /*----------------------------------------------------------------------------
  * PROGRAM SETTINGS
  *****************************************************************************/
  // The number of elements ca be given as 1st argument (Default is 65536).
  dim_data = ( argc == 1 ) ? MAX_NUM_ELEMENTS : atoi(argv[1]);

  if ( dim_data > MAX_NUM_ELEMENTS || !isPowerOfTwo(dim_data) ) {
    fprintf(stderr, "[ps-cuda]> Error: the number of elements must be a power of 2 and <= 65536, but you entered %d\n", dim_data);
    return EXIT_FAILURE;
  }


  /*----------------------------------------------------------------------------
  * HEADER
  *****************************************************************************/
  #ifndef PRINT_OFF
  printf("*****************************************************************\n");
  printf("Prefix-Sum leveraging CUDA\n");
  printf("@Author:      Giacomo Marciani <gmarciani@acm.org>\n");
  printf("              Gabriele Santi   <gsanti@acm.org>\n");
  printf("@Institution: University of Rome Tor Vergata\n");
  printf("@Credits:     Massimo Bernaschi\n");
  printf("*****************************************************************\n");
  printf("                     ---   Input   ---                           \n");
  printf("Number of elements: %d\n", dim_data);
  printf("                     ---   Grid   ---                           \n");
  printf("Block Dimension:    %d\n", BLOCK_SIZE);
  printf("Warp Size:          %d\n", WARP_SIZE);
  printf("-----------------------------------------------------------------\n");
  #ifdef VERBOSE
  printGpuInfo();
  printf("-----------------------------------------------------------------\n");
  #endif
  printf("-----------------------------------------------------------------\n");
  #endif

  HANDLE_ERROR(cudaSetDevice(0));

  /*----------------------------------------------------------------------------
    RESOURCES ALLOCATION AND INITIALIZATION
  *****************************************************************************/

  /*---------------------------------------------------------------------------+
  |  Host Allocations and Initializations:
  |     * input: input data, array of integers.
  |     * output: output data, array of integers.
  +---------------------------------------------------------------------------*/
  // Sizes
  const size_t size_data =  sizeof(int) * dim_data; // bytes in data

  // Allocations
  HANDLE_ERROR(cudaMallocHost((void **)&input, size_data));
  HANDLE_ERROR(cudaMallocHost((void **)&output, size_data));


  /*----------------------------------------------------------------------------
  * PROCESSING
  *****************************************************************************/
  #ifndef PRINT_OFF
  printf("[ps-cuda]> Processing\n");
  #endif
  
  prefix_sum(input, output, dim_data);

  /*----------------------------------------------------------------------------
  * CORRECTNESS CHECK
  *****************************************************************************/
  #ifdef CHECK_CORRECTNESS
  const bool check = isPrefixSumCorrect(input, output, dim_data);
  printf("%s\n", (check) ? "CORRECT" : "ERROR");
  #endif


  /*----------------------------------------------------------------------------
  * FREE RESOURCES (HOST)
  *****************************************************************************/
  HANDLE_ERROR(cudaFreeHost(input));
  HANDLE_ERROR(cudaFreeHost(output));

  return EXIT_SUCCESS;
}
