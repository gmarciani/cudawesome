/*
 * @Name: mainutil.h
 * @Description: Utilities form main.
 *
 * @Author: Giacomo Marciani <gmarciani@acm.org>
 *          Gabriele Santi   <gsanti@acm.org>
 *
 * @Institution: University of Rome Tor Vergata
 */

#ifndef __MAINUTIL_H__
#define __MAINUTIL_H__

#include <stdio.h>

// Verbosity specification
#ifdef VERBOSITY_ON
#define VERBOSE
#else
#undef VERBOSE
#endif

// Profile specification
#ifdef PROFILE_KERNEL_ON
#define PROFILE_KERNEL
#else
#undef PROFILE_KERNEL
#endif

// Correctness check specification
#ifdef CHECK_CORRECTNESS_ON
#define CHECK_CORRECTNESS
#else
#undef CHECK_CORRECTNESS
#endif


/*------------------------------------------------------------------------------
   @description   Print GPUs information
   @return  void
   ---------------------------------------------------------------------------*/
void printGpuInfo(void);

/*------------------------------------------------------------------------------
  @description      Verifies the prefix sum result.
  @param   input    Input data.
  @param   output   Output data.
  @param   n        Number of elements.
  @return  True, if the result is correct; False, otherwise.
  ----------------------------------------------------------------------------*/
bool isPrefixSumCorrect(int *input, int *output, int n);

/*------------------------------------------------------------------------------
  @description  Test if the given number is a power of two.
  @param   x    The number to test.
  @return  True, if the given number is a power of two; False, otherwise.
  ----------------------------------------------------------------------------*/
bool isPowerOfTwo(int x);

#endif  // __MAINUTIL_H__
