/*
 * @Name: precision.h
 * @Description: Common utilities for precision management.
 *
 * @Author: Giacomo Marciani <gmarciani@acm.org>
 * @Institution: University of Rome Tor Vergata
 */

#ifndef __PRECISION_H__
#define __PRECISION_H__

static void matrixDelta(int *a, int *b, int dimX, int dimY, double epsilon, int *cntErrors) {
  *cntErrors = 0;
  for (int y = 0; y < dimY; y++) {
    for (int x = 0; x < dimX; x++) {
      if (fabs(a[y * dimX + x], b[y * dimX + x]) > epsilon) {
        *cntErrors += 1;
      }
    }
  }
}

#endif  // __PRECISION_H__
