/*
 * @Name: ps.cuh
 * @Description: CUDA kernels and wrappers for Prefix-Sum operations.
 *
 * @Author: Giacomo Marciani <gmarciani@acm.org>
 *          Gabriele Santi   <gsanti@acm.org>
 *
 * @Institution: University of Rome Tor Vergata
 */

#ifndef __PS_CUH__
#define __PS_CUH__

#include <stdlib.h>
#include <stdio.h>


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
__host__ void prefix_sum(int *input, int *output, int dim_data);


/*------------------------------------------------------------------------------
 KERNELS
*******************************************************************************/

/*------------------------------------------------------------------------------
  @description  Computes the Prefix-Sum on the input array.
  @param   data         Input/Outut data (output is overwritten).
  @param   partialSums  Partial sums, shared between blocks.
  @param   warpSize     The warp size.
  @return  void.
  ----------------------------------------------------------------------------*/
__global__ void __prefix_sum(int *data, int *partialSums, int warpSize);


#endif // __PS_CUH__
