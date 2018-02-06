/*
 * @Name: error.h
 * @Description: Common utilities for error management.
 *
 *
 */
#ifndef __ERROR_H__
#define __ERROR_H__


#include <stdio.h>

static void handleError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
    printf("%s in %s at line %d\n",
    cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}

#define HANDLE_ERROR(err) (handleError(err, __FILE__, __LINE__))

#endif  // __ERROR_H__
