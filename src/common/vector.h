/*
 * @Name: matrix.h
 * @Description: Common utilities for vector calculus.
 *
 * @Author: Giacomo Marciani <gmarciani@acm.org>
 * @Institution: University of Rome Tor Vergata
 */

#ifndef __VECTOR_H__
#define __VECTOR_H__

static void vector_add(int *a, int *b, int *c, int dim) {
  for (int i = 0; i < dim; i++) {
    c[i] = a[i] + b[i];
  }
}

static void vector_dot(int *a, int *b, int *c, int dim) {
  for (int i = 0; i < dim; i++) {
    *c += a[i] * b[i];
  }
}

#endif  // __VECTOR_H__
