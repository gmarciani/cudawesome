/*
 * @Name: psutil.cuh
 * @Description: Utilities
 *
 */

#ifndef __PSUTIL_CUH__
#define __PSUTIL_CUH__

#include <stdlib.h>
#include <stdio.h>

// The block size for the whole application.
#if defined(BLOCK_DIM_4)
#define BLOCK_SIZE 4
#elif defined(BLOCK_DIM_8)
#define BLOCK_SIZE 8
#elif defined(BLOCK_DIM_16)
#define BLOCK_SIZE 16
#elif defined(BLOCK_DIM_32)
#define BLOCK_SIZE 32
#elif defined(BLOCK_DIM_64)
#define BLOCK_SIZE 64
#elif defined(BLOCK_DIM_128)
#define BLOCK_SIZE 128
#elif defined(BLOCK_DIM_256)
#define BLOCK_SIZE 256
#elif defined(BLOCK_DIM_512)
#define BLOCK_SIZE 512
#elif defined(BLOCK_DIM_1024)
#define BLOCK_SIZE 1024
#else
#define BLOCK_SIZE 32
#endif

// The warp size
#define WARP_SIZE 32

// Array size declaration
/*
#if defined(N_ELEMENTS_4)
#define N_ELEMENTS 8
#elif defined(N_ELEMENTS_8)
#define N_ELEMENTS 8
#elif defined(N_ELEMENTS_16)
#define N_ELEMENTS 16
#elif defined(N_ELEMENTS_32)
#define N_ELEMENTS 32
#elif defined(N_ELEMENTS_64)
#define N_ELEMENTS 64
#elif defined(N_ELEMENTS_128)
#define N_ELEMENTS 128
#elif defined(N_ELEMENTS_256)
#define N_ELEMENTS 256
#elif defined(N_ELEMENTS_512)
#define N_ELEMENTS 512
#elif defined(N_ELEMENTS_1024)
#define N_ELEMENTS 1024
#elif defined(N_ELEMENTS_2048)
#define N_ELEMENTS 2048
#elif defined(N_ELEMENTS_4096)
#define N_ELEMENTS 4096
#elif defined(N_ELEMENTS_8192)
#define N_ELEMENTS 8192
#elif defined(N_ELEMENTS_16384)
#define N_ELEMENTS 16384
#elif defined(N_ELEMENTS_32768)
#define N_ELEMENTS 32768
#elif defined(N_ELEMENTS_65536)
#define N_ELEMENTS 65536
#else
#define N_ELEMENTS 65536
#endif
*/

#endif // __PSUTIL_CUH__
