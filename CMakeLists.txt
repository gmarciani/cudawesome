cmake_minimum_required(VERSION 3.0)

project(cudawesome)

find_package(CUDA QUIET REQUIRED)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

##
# BASIC
##
set(HELLO_WORLD_SOURCE_FILES src/basic/hello_world.cu)
cuda_add_executable(hello_world ${HELLO_WORLD_SOURCE_FILES})

##
# BASIC/INTEGER
##
set(INTEGER_ADD_SOURCE_FILES src/basic/integer_add.cu
src/common/error.h)
cuda_add_executable(integer_add ${INTEGER_ADD_SOURCE_FILES})

set(INTEGER_ADD_PTR_SOURCE_FILES src/basic/integer_add_ptr.cu
src/common/error.h)
cuda_add_executable(integer_add_ptr ${INTEGER_ADD_PTR_SOURCE_FILES})

##
# BASIC/MATRIX
##
set(MATRIX_ADD_NXM_SOURCE_FILES src/basic/matrix/matrix_add_nxm.cu
src/common/error.h src/common/random.h src/common/matrix.h)
cuda_add_executable(matrix_add_nxm ${MATRIX_ADD_NXM_SOURCE_FILES})

set(MATRIX_ADD_NXN_SOURCE_FILES src/basic/matrix/matrix_add_nxn.cu
src/common/error.h src/common/random.h src/common/matrix.h)
cuda_add_executable(matrix_add_nxn ${MATRIX_ADD_NXN_SOURCE_FILES})

set(MATRIX_MUL_NXM_SOURCE_FILES src/basic/matrix/matrix_mul_nxm.cu
src/common/error.h src/common/random.h src/common/matrix.h)
cuda_add_executable(matrix_mul_nxm ${MATRIX_MUL_NXM_SOURCE_FILES})

set(MATRIX_MUL_NXN_SOURCE_FILES src/basic/matrix/matrix_mul_nxn.cu
src/common/error.h src/common/random.h src/common/matrix.h)
cuda_add_executable(matrix_mul_nxn ${MATRIX_MUL_NXN_SOURCE_FILES})

##
# BASIC/VECTOR
##
set(VECTOR_ADD_BLOCKS_THREADS_SOURCE_FILES src/basic/vector/vector_add_blocks_threads.cu
src/common/error.h src/common/random.h src/common/vector.h)
cuda_add_executable(vector_add_blocks_threads ${VECTOR_ADD_BLOCKS_THREADS_SOURCE_FILES})

set(VECTOR_ADD_BLOCKS_SOURCE_FILES src/basic/vector/vector_add_blocks.cu
src/common/error.h src/common/random.h src/common/vector.h)
cuda_add_executable(vector_add_blocks ${VECTOR_ADD_BLOCKS_SOURCE_FILES})

set(VECTOR_ADD_THREADS_SOURCE_FILES src/basic/vector/vector_add_threads.cu
src/common/error.h src/common/random.h src/common/vector.h)
cuda_add_executable(vector_add_threads ${VECTOR_ADD_THREADS_SOURCE_FILES})

set(VECTOR_ADD_SOURCE_FILES src/basic/vector/vector_add.cu
src/common/error.h src/common/random.h src/common/vector.h)
cuda_add_executable(vector_add ${VECTOR_ADD_SOURCE_FILES})

set(VECTOR_DOT_BLOCKS_THREADS_SOURCE_FILES src/basic/vector/vector_dot_blocks_threads.cu
src/common/error.h src/common/random.h src/common/vector.h)
cuda_add_executable(vector_dot_blocks_threads ${VECTOR_DOT_BLOCKS_THREADS_SOURCE_FILES})

set(VECTOR_DOT_THREADS_SOURCE_FILES src/basic/vector/vector_dot_threads.cu
src/common/error.h src/common/random.h src/common/vector.h)
cuda_add_executable(vector_dot_threads ${VECTOR_DOT_THREADS_SOURCE_FILES})

set(VECTOR_DOT_SOURCE_FILES src/basic/vector/vector_dot.cu
src/common/error.h src/common/random.h src/common/vector.h)
cuda_add_executable(vector_dot ${VECTOR_DOT_SOURCE_FILES})

##
# EVENT
##
set(EVENT_MATRIX_MUL_SOURCE_FILES src/event/event_matrix_mul.cu
src/common/error.h src/common/random.h src/common/matrix.h)
cuda_add_executable(event_matrix_mul ${EVENT_MATRIX_MUL_SOURCE_FILES})

##
# INFO
##
set(GPU_INFO_SOURCE_FILES src/info/gpu_info.cu
src/common/error.h)
cuda_add_executable(gpu_info ${GPU_INFO_SOURCE_FILES})
