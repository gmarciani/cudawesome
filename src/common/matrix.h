/*
 * @Name: matrix.h
 * @Description: Common utilities for matrix calculus.
 *
 * @Author: Giacomo Marciani <gmarciani@acm.org>
 * @Institution: University of Rome Tor Vergata
 */

#ifndef __MATRIX_H__
#define __MATRIX_H__

static void matrix_add(int *a, int *b, int *c, int dimX, int dimY) {
  for (int y = 0; y < dimY; y++) {
    for (int x = 0; x < dimX; x++) {
      int idx = y * dimX + x:
      c[idx] = a[idx] + b[idx];
    }
  }
}

static void matrix_mul(int *a, int *b, int *c, int dimX1Y2, int dimY1, int dimX2) {
  for (int y = 0; y < dimY1; y++) {
    for (int x = 0; x < dimX2; x++) {
      int val = 0;
      for (int k = 0; k < dimX1Y2; k++) {
        val += a[y * dimX2 + k] * b[k * dimX1Y2 + x];
      }
      c[y * dimX1 + x] = val;
    }
  }
}

#endif  // __MATRIX_H__
