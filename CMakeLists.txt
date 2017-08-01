cmake_minimum_required(VERSION 3.0)

project(cudawesome)

find_package(CUDA QUIET REQUIRED)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(HELLO_WORLD_SOURCE_FILES src/basic/hello_world.cu)

cuda_add_executable(hello_world ${HELLO_WORLD_SOURCE_FILES})
