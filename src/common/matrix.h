/*
 * @Name: matrix.h
 * @Description: Common utilities for matrix calculus.
 *
 * @Author: Giacomo Marciani <gmarciani@acm.org>
 * @Institution: University of Rome Tor Vergata
 */

#ifndef __MATRIX_H__
#define __MATRIX_H__

static void matrix_add(double *a, double *b, double *c, int dimX, int dimY) {
  for (int y = 0; y < dimY; y++) {
    for (int x = 0; x < dimX; x++) {
      int i = y * dimX + x;
      c[i] = a[i] + b[i];
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
      c[y * dimX1Y2 + x] = val;
    }
  }
}

static void matrix_print_double(char *name, double *a, int dimX, int dimY) {
  printf("%s=[\n", name);
  for (int i = 0; i < dimX * dimY; i++) {
    printf("%f ", a[i]);
  }
  printf("]\n");
}

#endif  // __MATRIX_H__
