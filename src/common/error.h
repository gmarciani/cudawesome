/*
 * @Name: error.h
 * @Description: Common utilities for error management.
 *
 * @Author: Giacomo Marciani <gmarciani@acm.org>
 * @Institution: University of Rome Tor Vergata
 */

#ifndef __ERROR_H__
#define __ERROR_H__

static void handleError(cudaError_t err, const char *file, const int line) {
  if (err != cudaSuccess) {
    fprintf(stderr, "%s in %s at line %d\n",
    cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}

#define HANDLE_ERROR(err) (handleError(err, __FILE__, __LINE__))

#define HANDLE_NULL(a) {if ((a) == NULL) { \
                            fprintf(stderr, "Host memory failed in %s at line %d\n", \
                                    __FILE__, __LINE__); \
                            exit( EXIT_FAILURE );}}

#endif  // __ERROR_H__
