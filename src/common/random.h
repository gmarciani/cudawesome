/*
 * @Name: random.h
 * @Description: Common utilities for random sequence generation.
 *
 * @Author: Giacomo Marciani <gmarciani@acm.org>
 * @Institution: University of Rome Tor Vergata
 */

#ifndef __RANDOM_H__
#define __RANDOM_H__

static void random_matrix_double(double *a, int rows, int cols) {
  for (int i = 0; i < rows * cols; i++) {
    a[i] = rand();
  }
}

static void random_matrix_float(float *a, int rows, int cols) {
  for (int i = 0; i < rows * cols; i++) {
    a[i] = rand();
  }
}

static void random_matrix_int(int *a, int rows, int cols) {
  for (int i = 0; i < rows * cols; i++) {
    a[i] = rand();
  }
}

static void random_matrix_long(long *a, int rows, int cols) {
  for (int i = 0; i < rows * cols; i++) {
    a[i] = rand();
  }
}

static void random_matrix_short(short *a, int rows, int cols) {
  for (int i = 0; i < rows * cols; i++) {
    a[i] = rand();
  }
}

static void random_vector_double(double *a, int n) {
  for (int i = 0; i < n; i++) {
    a[i] = rand();
  }
}

static void random_vector_float(float *a, int n) {
  for (int i = 0; i < n; i++) {
    a[i] = rand();
  }
}

static void random_vector_int(int *a, int n) {
  for (int i = 0; i < n; i++) {
    a[i] = rand();
  }
}

static void random_vector_long(long *a, int n) {
  for (int i = 0; i < n; i++) {
    a[i] = rand();
  }
}

static void random_vector_short(short *a, int n) {
  for (int i = 0; i < n; i++) {
    a[i] = rand();
  }
}

#endif  // __RANDOM_H__
