CC=nvcc

CFLAGS= --machine 64 --gpu-architecture=sm_35 --compiler-options -Wall,-Wno-unused-function

MFLAGS= --define-macro FLOAT

OFLAGS= --optimize 3

DFLAGS= --debug --device-debug

PFLAGS= --profile

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
	$(CC) $(CFLAGS) $(MFLAGS) $(OFLAGS) -o $(BIN)/$@ $^

##
# basic
##
basic: hello_world integer vector matrix

hello_world: $(BASIC_DIR)/hello_world.cu
	$(CC) $(CFLAGS) $(MFLAGS) $(OFLAGS) -o $(BIN)/$@ $^

##
# basic/integer
##
integer: integer_add_ptr integer_add

integer_add_ptr: $(INTEGER_DIR)/integer_add_ptr.cu
	$(CC) $(CFLAGS) $(MFLAGS) $(OFLAGS) -o $(BIN)/$@ $^

integer_add: $(INTEGER_DIR)/integer_add.cu
	$(CC) $(CFLAGS) $(MFLAGS) $(OFLAGS) -o $(BIN)/$@ $^

##
# basic/matrix
##
matrix: all_matrix_add all_matrix_mul

all_matrix_add: matrix_add_nxm matrix_add_nxn

all_matrix_mul: matrix_mul_nxm matrix_mul_nxn

matrix_add_nxm: $(MATRIX_DIR)/matrix_add_nxm.cu
	$(CC) $(CFLAGS) $(MFLAGS) $(OFLAGS) -o $(BIN)/$@ $^

matrix_add_nxn: $(MATRIX_DIR)/matrix_add_nxn.cu
	$(CC) $(CFLAGS) $(MFLAGS) $(OFLAGS) -o $(BIN)/$@ $^

matrix_mul_nxm: $(MATRIX_DIR)/matrix_mul_nxm.cu
	$(CC) $(CFLAGS) $(MFLAGS) $(OFLAGS) -o $(BIN)/$@ $^

matrix_mul_nxn: $(MATRIX_DIR)/matrix_mul_nxn.cu
	$(CC) $(CFLAGS) $(MFLAGS) $(OFLAGS) -o $(BIN)/$@ $^

##
# basic/vector
##
vector: all_vector_add all_vector_dot

all_vector_add: all_vector_add_float all_vector_add_int

all_vector_add_float: vector_add_float

all_vector_add_int: vector_add_int

all_vector_dot: all_vector_dot_float all_vector_dot_int

all_vector_dot_float: vector_dot_float_1 vector_dot_float_2 vector_dot_float_3

all_vector_dot_int: vector_dot_int_1 vector_dot_int_2 vector_dot_int_3

vector_add_float: $(VECTOR_DIR)/vector_add_float.cu
	$(CC) $(CFLAGS) $(MFLAGS) $(OFLAGS) -o $(BIN)/$@ $^

vector_add_int: $(VECTOR_DIR)/vector_add_int.cu
	$(CC) $(CFLAGS) $(MFLAGS) $(OFLAGS) -o $(BIN)/$@ $^

vector_dot_float_1: $(VECTOR_DIR)/vector_dot_float_1.cu
	$(CC) $(CFLAGS) $(MFLAGS) $(OFLAGS) -o $(BIN)/$@ $^

vector_dot_float_2: $(VECTOR_DIR)/vector_dot_float_2.cu
	$(CC) $(CFLAGS) $(MFLAGS) $(OFLAGS) -o $(BIN)/$@ $^

vector_dot_float_3: $(VECTOR_DIR)/vector_dot_float_3.cu
	$(CC) $(CFLAGS) $(MFLAGS) $(OFLAGS) -o $(BIN)/$@ $^

vector_dot_int_1: $(VECTOR_DIR)/vector_dot_int_1.cu
	$(CC) $(CFLAGS) $(MFLAGS) $(OFLAGS) -o $(BIN)/$@ $^

vector_dot_int_2: $(VECTOR_DIR)/vector_dot_int_2.cu
	$(CC) $(CFLAGS) $(MFLAGS) $(OFLAGS) -o $(BIN)/$@ $^

vector_dot_int_3: $(VECTOR_DIR)/vector_dot_int_3.cu
	$(CC) $(CFLAGS) $(MFLAGS) $(OFLAGS) -o $(BIN)/$@ $^

##
# clean
##
clean:
	rm -rf $(BIN)
