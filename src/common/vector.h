/*
 * @Name: matrix.h
 * @Description: Common utilities for vector calculus.
 *
 * @Author: Giacomo Marciani <gmarciani@acm.org>
 * @Institution: University of Rome Tor Vergata
 */

#ifndef __VECTOR_H__
#define __VECTOR_H__

static void vector_add_double(const double *a, const double *b, double *c, const unsigned int dim) {
  for (unsigned int i = 0; i < dim; i++) {
    c[i] = a[i] + b[i];
  }
}

static void vector_add_float(const float *a, const float *b, float *c, const unsigned int dim) {
  for (unsigned int i = 0; i < dim; i++) {
    c[i] = a[i] + b[i];
  }
}

static void vector_add_int(const int *a, const int *b, int *c, const unsigned int dim) {
  for (unsigned int i = 0; i < dim; i++) {
    c[i] = a[i] + b[i];
  }
}

static void vector_add_long(const long *a, const long *b, long *c, const unsigned int dim) {
  for (unsigned int i = 0; i < dim; i++) {
    c[i] = a[i] + b[i];
  }
}

static void vector_add_short(const short *a, const short *b, short *c, const unsigned int dim) {
  for (unsigned int i = 0; i < dim; i++) {
    c[i] = a[i] + b[i];
  }
}

static void vector_dot_double(const double *a, const double *b, double *c, const unsigned int dim) {
  *c = 0.0;
  for (unsigned int i = 0; i < dim; i++) {
    *c += a[i] * b[i];
  }
}

static void vector_dot_float(const float *a, const float *b, float *c, const unsigned int dim) {
  *c = 0.0f;
  for (unsigned int i = 0; i < dim; i++) {
    *c += a[i] * b[i];
  }
}

static void vector_dot_int(const int *a, const int *b, int *c, const unsigned int dim) {
  *c = 0;
  for (unsigned int i = 0; i < dim; i++) {
    *c += a[i] * b[i];
  }
}

static void vector_dot_long(const long *a, const long *b, long *c, const unsigned int dim) {
  *c = 0;
  for (unsigned int i = 0; i < dim; i++) {
    *c += a[i] * b[i];
  }
}

static void vector_dot_short(const short *a, const short *b, short *c, const unsigned int dim) {
  *c = 0;
  for (unsigned int i = 0; i < dim; i++) {
    *c += a[i] * b[i];
  }
}

static bool vector_equals_double(const double *actual, const double *expected, const unsigned int dim) {
  for (unsigned int i = 0; i < dim; i++) {
    if (actual[i] != expected[i]) {
      return false;
    }
  }
  return true;
}

static bool vector_equals_float(const float *actual, const float *expected, const unsigned int dim) {
  for (unsigned int i = 0; i < dim; i++) {
    if (actual[i] != expected[i]) {
      return false;
    }
  }
  return true;
}

static bool vector_equals_int(const int *actual, const int *expected, const unsigned int dim) {
  for (unsigned int i = 0; i < dim; i++) {
    if (actual[i] != expected[i]) {
      return false;
    }
  }
  return true;
}

static bool vector_equals_long(const long *actual, const long *expected, const unsigned int dim) {
  for (unsigned int i = 0; i < dim; i++) {
    if (actual[i] != expected[i]) {
      return false;
    }
  }
  return true;
}

static bool vector_equals_short(const short *actual, const short *expected, const unsigned int dim) {
  for (unsigned int i = 0; i < dim; i++) {
    if (actual[i] != expected[i]) {
      return false;
    }
  }
  return true;
}

static bool vector_equals_err_double(const double *actual, const double *expected, const unsigned int dim, const float err) {
  for (unsigned int i = 0; i < dim; i++) {
    if (fabs(expected[i] - actual[i]) > err * expected[i]) {
      return false;
    }
  }
  return true;
}

static bool vector_equals_err_float(const float *actual, const float *expected, const unsigned int dim, const float err) {
  for (unsigned int i = 0; i < dim; i++) {
    if (fabs(expected[i] - actual[i]) > err * expected[i]) {
      return false;
    }
  }
  return true;
}

static bool vector_equals_err_int(const int *actual, const int *expected, const unsigned int dim, const float err) {
  for (unsigned int i = 0; i < dim; i++) {
    if (abs(expected[i] - actual[i]) > err * expected[i]) {
      return false;
    }
  }
  return true;
}

static bool vector_equals_err_long(const long *actual, const long *expected, const unsigned int dim, const float err) {
  for (unsigned int i = 0; i < dim; i++) {
    if (abs(expected[i] - actual[i]) > err * expected[i]) {
      return false;
    }
  }
  return true;
}

static bool vector_equals_err_short(const short *actual, const short *expected, const unsigned int dim, const float err) {
  for (unsigned int i = 0; i < dim; i++) {
    if (abs(expected[i] - actual[i]) > err * expected[i]) {
      return false;
    }
  }
  return true;
}

static void vector_print_double(const char *name, const double *a, const unsigned int dim) {
  printf("%s=[\n", name);
  for (unsigned int i = 0; i < dim; i++) {
    printf("%lf ", a[i]);
  }
  printf("]\n");
}

static void vector_print_float(const char *name, const float *a, const unsigned int dim) {
  printf("%s=[\n", name);
  for (unsigned int i = 0; i < dim; i++) {
    printf("%f ", a[i]);
  }
  printf("]\n");
}

static void vector_print_int(const char *name, const int *a, const unsigned int dim) {
  printf("%s=[\n", name);
  for (unsigned int i = 0; i < dim; i++) {
    printf("%d ", a[i]);
  }
  printf("]\n");
}

static void vector_print_long(const char *name, const long *a, const unsigned int dim) {
  printf("%s=[\n", name);
  for (unsigned int i = 0; i < dim; i++) {
    printf("%ld ", a[i]);
  }
  printf("]\n");
}

static void vector_print_short(const char *name, const short *a, const unsigned int dim) {
  printf("%s=[\n", name);
  for (unsigned int i = 0; i < dim; i++) {
    printf("%d ", a[i]);
  }
  printf("]\n");
}

#endif  // __VECTOR_H__
