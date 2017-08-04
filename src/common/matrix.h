/*
 * @Name: matrix.h
 * @Description: Common utilities for matrix calculus.
 *
 * @Author: Giacomo Marciani <gmarciani@acm.org>
 * @Institution: University of Rome Tor Vergata
 */

#ifndef __MATRIX_H__
#define __MATRIX_H__

static void matrix_add_double(const double *a, const double *b, double *c, const unsigned int dimX, const unsigned int dimY) {
  for (unsigned int i = 0; i < dimX * dimY; i++) {
    c[i] = a[i] + b[i];
  }
}

static void matrix_add_float(const float *a, const float *b, float *c, const unsigned int dimX, const unsigned int dimY) {
  for (unsigned int i = 0; i < dimX * dimY; i++) {
    c[i] = a[i] + b[i];
  }
}

static void matrix_add_int(const int *a, const int *b, int *c, const unsigned int dimX, const unsigned int dimY) {
  for (unsigned int i = 0; i < dimX * dimY; i++) {
    c[i] = a[i] + b[i];
  }
}

static void matrix_add_long(const long *a, const long *b, long *c, const unsigned int dimX, const unsigned int dimY) {
  for (unsigned int i = 0; i < dimX * dimY; i++) {
    c[i] = a[i] + b[i];
  }
}

static void matrix_add_short(const short *a, const short *b, short *c, const unsigned int dimX, const unsigned int dimY) {
  for (unsigned int i = 0; i < dimX * dimY; i++) {
    c[i] = a[i] + b[i];
  }
}

static bool matrix_equals_double(const double *actual, const double *expected, const unsigned int dimX, const unsigned int dimY) {
  for (unsigned int i = 0; i < dimX * dimY; i++) {
    if (actual[i] != expected[i]) {
      return false;
    }
  }
  return true;
}

static bool matrix_equals_float(const float *actual, const float *expected, const unsigned int dimX, const unsigned int dimY) {
  for (unsigned int i = 0; i < dimX * dimY; i++) {
    if (actual[i] != expected[i]) {
      return false;
    }
  }
  return true;
}

static bool matrix_equals_int(const int *actual, const int *expected, const unsigned int dimX, const unsigned int dimY) {
  for (unsigned int i = 0; i < dimX * dimY; i++) {
    if (actual[i] != expected[i]) {
      return false;
    }
  }
  return true;
}

static bool matrix_equals_long(const long *actual, const long *expected, const unsigned int dimX, const unsigned int dimY) {
  for (unsigned int i = 0; i < dimX * dimY; i++) {
    if (actual[i] != expected[i]) {
      return false;
    }
  }
  return true;
}

static bool matrix_equals_short(const short *actual, const short *expected, const unsigned int dimX, const unsigned int dimY) {
  for (unsigned int i = 0; i < dimX * dimY; i++) {
    if (actual[i] != expected[i]) {
      return false;
    }
  }
  return true;
}

static void matrix_mul_double(const double *a, const double *b, double *c, const unsigned int dimX1Y2, const unsigned int dimY1, const unsigned int dimX2) {
  for (unsigned int y = 0; y < dimY1; y++) {
    for (unsigned int x = 0; x < dimX2; x++) {
      double val = 0.0;
      for (unsigned int k = 0; k < dimX1Y2; k++) {
        val += a[y * dimX2 + k] * b[k * dimX1Y2 + x];
      }
      c[y * dimX1Y2 + x] = val;
    }
  }
}

static void matrix_mul_float(const float *a, const float *b, float *c, const unsigned int dimX1Y2, const unsigned int dimY1, const unsigned int dimX2) {
  for (unsigned int y = 0; y < dimY1; y++) {
    for (unsigned int x = 0; x < dimX2; x++) {
      float val = 0.0;
      for (unsigned int k = 0; k < dimX1Y2; k++) {
        val += a[y * dimX2 + k] * b[k * dimX1Y2 + x];
      }
      c[y * dimX1Y2 + x] = val;
    }
  }
}

static void matrix_mul_int(const int *a, const int *b, int *c, const unsigned int dimX1Y2, const unsigned int dimY1, const unsigned int dimX2) {
  for (unsigned int y = 0; y < dimY1; y++) {
    for (unsigned int x = 0; x < dimX2; x++) {
      int val = 0;
      for (unsigned int k = 0; k < dimX1Y2; k++) {
        val += a[y * dimX2 + k] * b[k * dimX1Y2 + x];
      }
      c[y * dimX1Y2 + x] = val;
    }
  }
}

static void matrix_mul_long(const long *a, const long *b, long *c, const unsigned int dimX1Y2, const unsigned int dimY1, const unsigned int dimX2) {
  for (unsigned int y = 0; y < dimY1; y++) {
    for (unsigned int x = 0; x < dimX2; x++) {
      long val = 0;
      for (unsigned int k = 0; k < dimX1Y2; k++) {
        val += a[y * dimX2 + k] * b[k * dimX1Y2 + x];
      }
      c[y * dimX1Y2 + x] = val;
    }
  }
}

static void matrix_mul_short(const short *a, const short *b, short *c, const unsigned int dimX1Y2, const unsigned int dimY1, const unsigned int dimX2) {
  for (unsigned int y = 0; y < dimY1; y++) {
    for (unsigned int x = 0; x < dimX2; x++) {
      short val = 0;
      for (unsigned int k = 0; k < dimX1Y2; k++) {
        val += a[y * dimX2 + k] * b[k * dimX1Y2 + x];
      }
      c[y * dimX1Y2 + x] = val;
    }
  }
}

static void matrix_print_double(const char *name, const double *a, const unsigned int dimX, const unsigned int dimY) {
  printf("%s=[\n", name);
  for (unsigned int i = 0; i < dimX * dimY; i++) {
    printf("%lf ", a[i]);
  }
  printf("]\n");
}

static void matrix_print_float(const char *name, const float *a, const unsigned int dimX, const unsigned int dimY) {
  printf("%s=[\n", name);
  for (unsigned int i = 0; i < dimX * dimY; i++) {
    printf("%f ", a[i]);
  }
  printf("]\n");
}

static void matrix_print_int(const char *name, const int *a, const unsigned int dimX, const unsigned int dimY) {
  printf("%s=[\n", name);
  for (unsigned int i = 0; i < dimX * dimY; i++) {
    printf("%d ", a[i]);
  }
  printf("]\n");
}

static void matrix_print_long(const char *name, const long *a, const unsigned int dimX, const unsigned int dimY) {
  printf("%s=[\n", name);
  for (unsigned int i = 0; i < dimX * dimY; i++) {
    printf("%ld ", a[i]);
  }
  printf("]\n");
}

static void matrix_print_short(const char *name, const short *a, const unsigned int dimX, const unsigned int dimY) {
  printf("%s=[\n", name);
  for (unsigned int i = 0; i < dimX * dimY; i++) {
    printf("%d ", a[i]);
  }
  printf("]\n");
}

#endif  // __MATRIX_H__
