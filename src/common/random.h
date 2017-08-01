/*
 * @Name: random.h
 * @Description: Common utilities for random sequence generation.
 *
 * @Author: Giacomo Marciani <gmarciani@acm.org>
 * @Institution: University of Rome Tor Vergata
 */

#ifndef __RANDOM_H__
#define __RANDOM_H__

static void random_ints(int *p, int n) {
  for(int i = 0; i < n; i++) {
    p[i] = rand();
  }
}

static void random_matrix_int(int *p, int rows, int cols) {
  for(int row = 0; i < rows; rows++) {
    for(int col = 0; col < cols; col++) {
      p[row * cols + col] = rand();
    }
  }
}

#endif  // __RANDOM_H__
