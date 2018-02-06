/*
 * @Name: matrix_mul_nxm_int.cu
 * @Description: Matrix (NxM) Integer Product.
 * Each matrix is viewed as a single block of memory.
 * Blocks and threads are viewed as a 2D grid.
 * Custom matrix dimension and block size.
 *
 * @Author: Giacomo Marciani <gmarciani@acm.org>
 * @Institution: University of Rome Tor Vergata
 *
 * @Usage: matrix_initadd_int matrixRows matrixCols matrixZ blockSize
 *
 * Default values:
 *  matrixDimX1: 4096
 *  matrixDimY1: 4096
 *  matrixDimX2: 4096
 *  matrixDimY2: 4096
 *  blockSize: 32
 */

#include <stdio.h>
#include <math.h>
#include "../../common/error.h"
#include "../../common/random.h"
#include "../../common/matrix.h"
#include "../../common/mathutil.h"


#define MAX_BLOCK_SIZE 1024

#define ALPHA 3

__global__ void matrixInitAdd(const int *A, const int *B, int *C, const int matrixRows, const int matrixCols, const int matrixZ) {
  extern __shared__ int shmem[];

  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int r = blockIdx.y * blockDim.y + threadIdx.y;
  const int z = blockIdx.z * blockDim.z + threadIdx.z;

  if (r >= matrixRows || c >= matrixCols || z >= matrixZ) return;

  const int tid = (threadIdx.z * blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

  const int pos = (r * matrixCols) + c; // A[r][c]
  const int pos2 = (z * matrixRows * matrixCols) + (r * matrixCols) + c; // B[z][r][c]

  if (z == 0) {
    C[pos] = ALPHA * A[pos];
  }

  int toAdd;
  if (r > 0) {
    toAdd = 2 * B[pos2];
  } else {
    toAdd = B[pos2];
  }

  shmem[tid] = toAdd;

  __syncthreads();

  if (threadIdx.z == 0) {
    int tot_toAdd = 0;
    for (int idxZ = 0; idxZ < blockDim.z; idxZ++) {
      const int pos_shmem = (idxZ * blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
      tot_toAdd += shmem[pos_shmem];
    }
    atomicAdd(C + pos, tot_toAdd);
  }
}

__host__ void gpuMatrixInitAdd(const int *A, const int *B, int *C,
    const int matrixRows, const int matrixCols, const int matrixZ,
    const dim3 gridDim, const dim3 blockDim) {
  int *dev_A, *dev_B, *dev_C; // device copies of A, B, C
  const size_t size_A = matrixRows * matrixCols * sizeof(int); // bytes for A
  const size_t size_B = matrixRows * matrixCols * matrixZ * sizeof(int); // bytes for B
  const size_t size_C = matrixRows * matrixCols * sizeof(int); // bytes for C
  const size_t size_shmem = sizeof(int) * blockDim.x * blockDim.y * blockDim.z;

  // allocate device copy of A, B, C
  HANDLE_ERROR(cudaMalloc((void**)&dev_A, size_A));
  HANDLE_ERROR(cudaMalloc((void**)&dev_B, size_B));
  HANDLE_ERROR(cudaMalloc((void**)&dev_C, size_C));

  // copy inputs to device
  HANDLE_ERROR(cudaMemcpy(dev_A, A, size_A, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_B, B, size_B, cudaMemcpyHostToDevice));

  // launch matrixInitAdd() kernel
  matrixInitAdd<<< gridDim, blockDim, size_shmem >>>(dev_A, dev_B, dev_C, matrixRows, matrixCols, matrixZ);

  // copy device result back to host copy of c
  HANDLE_ERROR(cudaMemcpy(C, dev_C, size_C, cudaMemcpyDeviceToHost));

  // free device
  HANDLE_ERROR(cudaFree(dev_A));
  HANDLE_ERROR(cudaFree(dev_B));
  HANDLE_ERROR(cudaFree(dev_C));
}

__host__ void cpuMatrixInitAdd(const int *A, const int *B, int *C,
  const int matrixRows, const int matrixCols, const int matrixZ) {

  int r, c, z, pos, pos2;

  for ( r = 0; r < matrixRows; r++ ) {
    for ( c = 0; c < matrixCols; c++ ) {
      pos = (r * matrixCols) + c; // A[r][c]
      C[pos] = ALPHA * A[pos];
      for ( z = 0; z < matrixZ; z++ ) {
        pos2 = (z * matrixRows * matrixCols) + (r * matrixCols) + c; // B[z][r][c]
        C[pos] += ( r > 0 ) ? 2 * B[pos2] : B[pos2];
      }
    }
  }
}

int main(const int argc, const char **argv) {
  int *A, *B, *C;         // host copies of A, B, C
  size_t size_A, size_B, size_C; // bytes for A, B, C
  int matrixRows, matrixCols, matrixZ; // matrices dimensions
  int blockSize; // block size
  cudaDeviceProp gpuInfo; // gpu properties

  // check arguments
  if (argc < 5) {
    fprintf(stderr, "Usage: %s matrixRows matrixCols matrixZ blockSize\n", argv[0]);
    exit(1);
  }

  matrixRows = atoi(argv[1]);
  matrixCols = atoi(argv[2]);
  matrixZ = atoi(argv[3]);
  blockSize = atoi(argv[4]);

  if (matrixRows < 1) {
    fprintf(stderr, "Error: matrixRows expected >= 1, got %d\n", matrixRows);
    exit(1);
  }

  if (matrixCols < 1) {
    fprintf(stderr, "Error: matrixCols expected >= 1, got %d\n", matrixCols);
    exit(1);
  }

  if (matrixZ < 1) {
    fprintf(stderr, "Error: matrixZ expected >= 1, got %d\n", matrixZ);
    exit(1);
  }

  if (blockSize < 1 || blockSize > MAX_BLOCK_SIZE) {
    fprintf(stderr, "Error: blockSize expected >= 1 and <= %d, got %d\n", MAX_BLOCK_SIZE, blockSize);
    exit(1);
  }

  // grid settings
  dim3 gridDim(1, 1, 1);
  dim3 blockDim(1, 1, 1);

  blockDim.x = pow(blockSize, 1/3.);
  blockDim.y = pow(blockSize, 1/3.);
  blockDim.z = pow(blockSize, 1/3.);

  gridDim.x = 1 + ((matrixCols - 1) / blockDim.x);
  gridDim.y = 1 + ((matrixRows - 1) / blockDim.y);
  gridDim.z = 1 + ((matrixZ - 1) / blockDim.z);

  size_A = matrixRows * matrixCols * sizeof(int);
  size_B = matrixRows * matrixCols * matrixZ * sizeof(int);
  size_C = matrixRows * matrixCols * sizeof(int);

  HANDLE_ERROR(cudaGetDeviceProperties(&gpuInfo, 0));

  printf("------------------------------------\n");
  printf("Matrix Integer Init-Add\n");
  printf("------------------------------------\n");
  printf("Matrix Dimension (A): %d x %d\n", matrixRows, matrixCols);
  printf("Matrix Dimension (B): %d x %d x %d\n", matrixRows, matrixCols, matrixZ);
  printf("Grid Size: (%d, %d, %d) (max: (%d, %d, %d))\n",
    gridDim.x, gridDim.y, gridDim.z,
    gpuInfo.maxGridSize[0], gpuInfo.maxGridSize[1], gpuInfo.maxGridSize[2]);
  printf("Block Size: (%d, %d, %d) (max: (%d, %d, %d))\n",
    blockDim.x, blockDim.y, blockDim.z,
    gpuInfo.maxThreadsDim[0], gpuInfo.maxThreadsDim[1], gpuInfo.maxThreadsDim[2]);
  printf("-----------------------------------\n");

  // allocate host copy of A, B, C
  HANDLE_NULL(A = (int*)malloc(size_A));
  HANDLE_NULL(B = (int*)malloc(size_B));
  HANDLE_NULL(C = (int*)malloc(size_C));

  // fill A, B with random data
  random_matrix_int_2(A, matrixRows, matrixCols);
  random_matrix_int_3(B, matrixRows, matrixCols, matrixZ);

  // launch kernel matrixInitAdd()
  gpuMatrixInitAdd(A, B, C, matrixRows, matrixCols, matrixZ, gridDim, blockDim);

  // test result
  int *EXPECTED;
  HANDLE_NULL(EXPECTED = (int*)malloc(size_C));
  cpuMatrixInitAdd(A, B, EXPECTED, matrixRows, matrixCols, matrixZ);
  const bool correct = matrix_equals_int(C, EXPECTED, matrixRows, matrixCols);
  if (!correct) {
    fprintf(stderr, "Error\n");
    matrix_pprint_int_2("A", A, matrixRows, matrixCols);
    matrix_pprint_int_3("B", B, matrixRows, matrixCols, matrixZ);
    matrix_pprint_int_2("C", C, matrixRows, matrixCols);
    matrix_pprint_int_2("EXPECTED", EXPECTED, matrixRows, matrixCols);
  } else {
    printf("Correct\n");
  }

  // free host
  free(A);
  free(B);
  free(C);
  free(EXPECTED);

  return 0;
}
