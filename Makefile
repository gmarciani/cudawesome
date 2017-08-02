CC=nvcc

CFLAGS= -m64

SRC=./src

BIN=./bin

.PHONY: all clean dir

all: dir basic info

##
# directories
##
BASIC_DIR=$(SRC)/basic
INTEGER_DIR=$(BASIC_DIR)/integer
MATRIX_DIR=$(BASIC_DIR)/matrix
VECTOR_DIR=$(BASIC_DIR)/vector
INFO_DIR=$(SRC)/info
dir:
	mkdir -p $(BIN)

##
# info
##
info: gpu_info

gpu_info: $(INFO_DIR)/gpu_info.cu
	$(CC) $(CFLAGS) -o $(BIN)/$@ $^

##
# basic
##
basic: hello_world integer vector matrix

hello_world: $(BASIC_DIR)/hello_world.cu
	$(CC) $(CFLAGS) -o $(BIN)/$@ $^

##
# basic/integer
##
integer: integer_add_ptr integer_add

integer_add_ptr: $(INTEGER_DIR)/integer_add_ptr.cu
	$(CC) $(CFLAGS) -o $(BIN)/$@ $^

integer_add: $(INTEGER_DIR)/integer_add.cu
	$(CC) $(CFLAGS) -o $(BIN)/$@ $^

##
# basic/matrix
##
matrix: all_matrix_add all_matrix_mul

all_matrix_add: matrix_add_nxm matrix_add_nxn

all_matrix_mul: matrix_mul_nxm matrix_mul_nxn

matrix_add_nxm: $(MATRIX_DIR)/matrix_add_nxm.cu
	$(CC) $(CFLAGS) -o $(BIN)/$@ $^

matrix_add_nxn: $(MATRIX_DIR)/matrix_add_nxn.cu
	$(CC) $(CFLAGS) -o $(BIN)/$@ $^

matrix_mul_nxm: $(MATRIX_DIR)/matrix_mul_nxm.cu
	$(CC) $(CFLAGS) -o $(BIN)/$@ $^

matrix_mul_nxn: $(MATRIX_DIR)/matrix_mul_nxn.cu
	$(CC) $(CFLAGS) -o $(BIN)/$@ $^
##
# basic/vector
##
vector: all_vector_add all_vector_dot

all_vector_add: vector_add_blocks_threads vector_add_blocks vector_add_threads vector_add

all_vector_dot: vector_dot_blocks_threads vector_dot_threads vector_dot

vector_add_blocks_threads: $(VECTOR_DIR)/vector_add_blocks_threads.cu
	$(CC) $(CFLAGS) -o $(BIN)/$@ $^

vector_add_blocks: $(VECTOR_DIR)/vector_add_blocks.cu
	$(CC) $(CFLAGS) -o $(BIN)/$@ $^

vector_add_threads: $(VECTOR_DIR)/vector_add_threads.cu
	$(CC) $(CFLAGS) -o $(BIN)/$@ $^

vector_add: $(VECTOR_DIR)/vector_add.cu
	$(CC) $(CFLAGS) -o $(BIN)/$@ $^

vector_dot_blocks_threads: $(VECTOR_DIR)/vector_dot_blocks_threads.cu
	$(CC) $(CFLAGS) -o $(BIN)/$@ $^

vector_dot_threads: $(VECTOR_DIR)/vector_dot_threads.cu
	$(CC) $(CFLAGS) -o $(BIN)/$@ $^

vector_dot: $(VECTOR_DIR)/vector_dot.cu
	$(CC) $(CFLAGS) -o $(BIN)/$@ $^

##
# clean
##
clean:
	rm -rf $(BIN)
