/*
 * @Name: matrix.h
 * @Description: Common utilities for vector calculus.
 *
 * @Author: Giacomo Marciani <gmarciani@acm.org>
 * @Institution: University of Rome Tor Vergata
 */

#ifndef __VECTOR_H__
#define __VECTOR_H__

static void vector_add_double(double *a, double *b, double *c, int dim) {
  for (int i = 0; i < dim; i++) {
    c[i] = a[i] + b[i];
  }
}

static void vector_add_float(float *a, float *b, float *c, int dim) {
  for (int i = 0; i < dim; i++) {
    c[i] = a[i] + b[i];
  }
}

static void vector_add_int(int *a, int *b, int *c, int dim) {
  for (int i = 0; i < dim; i++) {
    c[i] = a[i] + b[i];
  }
}

static void vector_add_long(long *a, long *b, long *c, int dim) {
  for (int i = 0; i < dim; i++) {
    c[i] = a[i] + b[i];
  }
}

static void vector_add_short(short *a, short *b, short *c, int dim) {
  for (int i = 0; i < dim; i++) {
    c[i] = a[i] + b[i];
  }
}

static void vector_dot_double(double *a, double *b, double *c, int dim) {
  *c = 0;
  for (int i = 0; i < dim; i++) {
    *c += a[i] * b[i];
  }
}

static void vector_dot_float(float *a, float *b, float *c, int dim) {
  *c = 0;
  for (int i = 0; i < dim; i++) {
    *c += a[i] * b[i];
  }
}

static void vector_dot_int(int *a, int *b, int *c, int dim) {
  *c = 0;
  for (int i = 0; i < dim; i++) {
    *c += a[i] * b[i];
  }
}

static void vector_dot_long(long *a, long *b, long *c, int dim) {
  *c = 0;
  for (int i = 0; i < dim; i++) {
    *c += a[i] * b[i];
  }
}

static void vector_dot_short(short *a, short *b, short *c, int dim) {
  *c = 0;
  for (int i = 0; i < dim; i++) {
    *c += a[i] * b[i];
  }
}

static bool vector_equals_double(double *actual, double *expected, int dim) {
  int i;
  for (i = 0; i < dim; i++) {
    if (actual[i] != expected[i]) {
      //fprintf(stderr, "Error: [%d] expected %f, got %f\n", i, expected[i], actual[i]);
      return false;
    }
  }
  return true;
}

static bool vector_equals_float(float *actual, float *expected, int dim) {
  int i;
  for (i = 0; i < dim; i++) {
    if (actual[i] != expected[i]) {
      //fprintf(stderr, "Error: [%d] expected %f, got %f\n", i, expected[i], actual[i]);
      return false;
    }
  }
  return true;
}

static bool vector_equals_int(int *actual, int *expected, int dim) {
  int i;
  for (i = 0; i < dim; i++) {
    if (actual[i] != expected[i]) {
      //fprintf(stderr, "Error: [%d] expected %d, got %d\n", i, expected[i], actual[i]);
      return false;
    }
  }
  return true;
}

static bool vector_equals_long(long *actual, long *expected, int dim) {
  int i;
  for (i = 0; i < dim; i++) {
    if (actual[i] != expected[i]) {
      //fprintf(stderr, "Error: [%d] expected %d, got %d\n", i, expected[i], actual[i]);
      return false;
    }
  }
  return true;
}

static bool vector_equals_short(short *actual, short *expected, int dim) {
  int i;
  for (i = 0; i < dim; i++) {
    if (actual[i] != expected[i]) {
      //fprintf(stderr, "Error: [%d] expected %d, got %d\n", i, expected[i], actual[i]);
      return false;
    }
  }
  return true;
}

#endif  // __VECTOR_H__
