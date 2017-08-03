/*
 * @Name: matrix.h
 * @Description: Common utilities for matrix calculus.
 *
 * @Author: Giacomo Marciani <gmarciani@acm.org>
 * @Institution: University of Rome Tor Vergata
 */

#ifndef __MATRIX_H__
#define __MATRIX_H__

static void matrix_add_double(double *a, double *b, double *c, int dimX, int dimY) {
  for (int y = 0; y < dimY; y++) {
    for (int x = 0; x < dimX; x++) {
      int i = y * dimX + x;
      c[i] = a[i] + b[i];
    }
  }
}

static void matrix_add_float(float *a, float *b, float *c, int dimX, int dimY) {
  for (int y = 0; y < dimY; y++) {
    for (int x = 0; x < dimX; x++) {
      int i = y * dimX + x;
      c[i] = a[i] + b[i];
    }
  }
}

static void matrix_add_int(int *a, int *b, int *c, int dimX, int dimY) {
  for (int y = 0; y < dimY; y++) {
    for (int x = 0; x < dimX; x++) {
      int i = y * dimX + x;
      c[i] = a[i] + b[i];
    }
  }
}

static void matrix_add_long(long *a, long *b, long *c, int dimX, int dimY) {
  for (int y = 0; y < dimY; y++) {
    for (int x = 0; x < dimX; x++) {
      int i = y * dimX + x;
      c[i] = a[i] + b[i];
    }
  }
}

static void matrix_add_short(short *a, short *b, short *c, int dimX, int dimY) {
  for (int y = 0; y < dimY; y++) {
    for (int x = 0; x < dimX; x++) {
      int i = y * dimX + x;
      c[i] = a[i] + b[i];
    }
  }
}

static bool matrix_equals_double(double *actual, double *expected, int dimX, int dimY) {
  int i;
  for (i = 0; i < dimX * dimY; i++) {
    if (actual[i] != expected[i]) {
      //fprintf(stderr, "Error: (%d,%d) expected %f, got %f\n", i % dimX, i - (i % dimX) / dimX, expected[i], actual[i]);
      return false;
    }
  }
  return true;
}

static bool matrix_equals_float(float *actual, float *expected, int dimX, int dimY) {
  int i;
  for (i = 0; i < dimX * dimY; i++) {
    if (actual[i] != expected[i]) {
      //fprintf(stderr, "Error: (%d,%d) expected %f, got %f\n", i % dimX, i - (i % dimX) / dimX, expected[i], actual[i]);
      return false;
    }
  }
  return true;
}

static bool matrix_equals_int(int *actual, int *expected, int dimX, int dimY) {
  int i;
  for (i = 0; i < dimX * dimY; i++) {
    if (actual[i] != expected[i]) {
      //fprintf(stderr, "Error: (%d,%d) expected %d, got %d\n", i % dimX, i - (i % dimX) / dimX, expected[i], actual[i]);
      return false;
    }
  }
  return true;
}

static bool matrix_equals_long(long *actual, long *expected, int dimX, int dimY) {
  int i;
  for (i = 0; i < dimX * dimY; i++) {
    if (actual[i] != expected[i]) {
      //fprintf(stderr, "Error: (%d,%d) expected %d, got %d\n", i % dimX, i - (i % dimX) / dimX, expected[i], actual[i]);
      return false;
    }
  }
  return true;
}

static bool matrix_equals_short(short *actual, short *expected, int dimX, int dimY) {
  int i;
  for (i = 0; i < dimX * dimY; i++) {
    if (actual[i] != expected[i]) {
      //fprintf(stderr, "Error: (%d,%d) expected %d, got %d\n", i % dimX, i - (i % dimX) / dimX, expected[i], actual[i]);
      return false;
    }
  }
  return true;
}

static void matrix_mul_double(double *a, double *b, double *c, int dimX1Y2, int dimY1, int dimX2) {
  for (int y = 0; y < dimY1; y++) {
    for (int x = 0; x < dimX2; x++) {
      double val = 0;
      for (int k = 0; k < dimX1Y2; k++) {
        val += a[y * dimX2 + k] * b[k * dimX1Y2 + x];
      }
      c[y * dimX1Y2 + x] = val;
    }
  }
}

static void matrix_mul_float(float *a, float *b, float *c, int dimX1Y2, int dimY1, int dimX2) {
  for (int y = 0; y < dimY1; y++) {
    for (int x = 0; x < dimX2; x++) {
      float val = 0;
      for (int k = 0; k < dimX1Y2; k++) {
        val += a[y * dimX2 + k] * b[k * dimX1Y2 + x];
      }
      c[y * dimX1Y2 + x] = val;
    }
  }
}

static void matrix_mul_int(int *a, int *b, int *c, int dimX1Y2, int dimY1, int dimX2) {
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

static void matrix_mul_long(long *a, long *b, long *c, int dimX1Y2, int dimY1, int dimX2) {
  for (int y = 0; y < dimY1; y++) {
    for (int x = 0; x < dimX2; x++) {
      long val = 0;
      for (int k = 0; k < dimX1Y2; k++) {
        val += a[y * dimX2 + k] * b[k * dimX1Y2 + x];
      }
      c[y * dimX1Y2 + x] = val;
    }
  }
}

static void matrix_mul_short(short *a, short *b, short *c, int dimX1Y2, int dimY1, int dimX2) {
  for (int y = 0; y < dimY1; y++) {
    for (int x = 0; x < dimX2; x++) {
      short val = 0;
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

static void matrix_print_float(char *name, float *a, int dimX, int dimY) {
  printf("%s=[\n", name);
  for (int i = 0; i < dimX * dimY; i++) {
    printf("%f ", a[i]);
  }
  printf("]\n");
}

static void matrix_print_int(char *name, int *a, int dimX, int dimY) {
  printf("%s=[\n", name);
  for (int i = 0; i < dimX * dimY; i++) {
    printf("%d ", a[i]);
  }
  printf("]\n");
}

static void matrix_print_long(char *name, long *a, int dimX, int dimY) {
  printf("%s=[\n", name);
  for (int i = 0; i < dimX * dimY; i++) {
    printf("%d ", a[i]);
  }
  printf("]\n");
}

static void matrix_print_short(char *name, short *a, int dimX, int dimY) {
  printf("%s=[\n", name);
  for (int i = 0; i < dimX * dimY; i++) {
    printf("%d ", a[i]);
  }
  printf("]\n");
}

#endif  // __MATRIX_H__
