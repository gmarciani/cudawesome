CC=nvcc

CFLAGS= --machine 64 --gpu-architecture=sm_35 --compiler-options -Wall,-Wno-unused-function -x cu --optimize 3 --define-macro FLOAT

DFLAGS= --debug --device-debug

PFLAGS= --profile

SRC=./src

BIN=./bin

.PHONY: all clean makedirs

all: makedirs basic info

##
# directories
##
BASIC_DIR=$(SRC)/basic
INTEGER_DIR=$(BASIC_DIR)/integer
MATRIX_DIR=$(BASIC_DIR)/matrix
VECTOR_DIR=$(BASIC_DIR)/vector
RAND_DIR=$(BASIC_DIR)/rand
INFO_DIR=$(SRC)/info
SCAFFOLDING_DIR=$(SRC)/scaffolding
FLOW_DIR=$(SRC)/flow

makedirs:
	mkdir -p $(BIN)

clean:
	rm -rf $(BIN)

##
# info
##
info: gpu_info

gpu_info: $(INFO_DIR)/gpu_info.cu
	$(CC) $(CFLAGS) -o $(BIN)/$@ $^

##
# scaffolding
##
hello_world: $(SCAFFOLDING_DIR)/hello_world.cu $(SCAFFOLDING_DIR)/include_cu/gpu_functions.cu $(SCAFFOLDING_DIR)/include_c/cpu_functions.c
	$(CC) $(CFLAGS) -o $(BIN)/$@ $^ --include-path $(SCAFFOLDING_DIR)


##
# flow
##
sync_async: $(FLOW_DIR)/sync_async.cu
	$(CC) $(CFLAGS) -o $(BIN)/$@ $^ --include-path $(FLOW_DIR)

##
# basic
##
basic: integer vector matrix rand

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
matrix: all_matrix_transfer all_matrix_add all_matrix_mul

all_matrix_transfer: matrix_transfer_2d matrix_transfer_3d

all_matrix_add: all_matrix_add_float all_matrix_add_int

all_matrix_add_float: matrix_add_nxm_float matrix_add_nxn_float

all_matrix_add_int: matrix_add_nxm_int matrix_add_nxn_int

all_matrix_mul: all_matrix_mul_float all_matrix_mul_int

all_matrix_mul_float: matrix_mul_nxm_float matrix_mul_nxn_float

all_matrix_mul_int: matrix_mul_nxm_int matrix_mul_nxn_int

matrix_transfer_2d: $(MATRIX_DIR)/matrix_transfer_2d.cu
	$(CC) $(CFLAGS) -o $(BIN)/$@ $^

matrix_transfer_3d: $(MATRIX_DIR)/matrix_transfer_3d.cu
	$(CC) $(CFLAGS) -o $(BIN)/$@ $^

matrix_add_nxm_float: $(MATRIX_DIR)/matrix_add_nxm_float.cu
	$(CC) $(CFLAGS) -o $(BIN)/$@ $^

matrix_add_nxn_float: $(MATRIX_DIR)/matrix_add_nxn_float.cu
	$(CC) $(CFLAGS) -o $(BIN)/$@ $^

matrix_add_nxm_int: $(MATRIX_DIR)/matrix_add_nxm_int.cu
	$(CC) $(CFLAGS) -o $(BIN)/$@ $^

matrix_add_nxn_int: $(MATRIX_DIR)/matrix_add_nxn_int.cu
	$(CC) $(CFLAGS) -o $(BIN)/$@ $^

matrix_mul_nxm_float: $(MATRIX_DIR)/matrix_mul_nxm_float.cu
	$(CC) $(CFLAGS) -o $(BIN)/$@ $^

matrix_mul_nxn_float: $(MATRIX_DIR)/matrix_mul_nxn_float.cu
	$(CC) $(CFLAGS) -o $(BIN)/$@ $^

matrix_mul_nxm_int: $(MATRIX_DIR)/matrix_mul_nxm_int.cu
	$(CC) $(CFLAGS) -o $(BIN)/$@ $^

matrix_mul_nxn_int: $(MATRIX_DIR)/matrix_mul_nxn_int.cu
	$(CC) $(CFLAGS) -o $(BIN)/$@ $^

matrix_initadd_int: $(MATRIX_DIR)/matrix_initadd_int.cu
	$(CC) $(CFLAGS) -o $(BIN)/$@ $^

##
# basic/vector
##
vector: all_vector_sum all_vector_dot

all_vector_sum: all_vector_sum_float all_vector_sum_int

all_vector_sum_float: vector_sum_float vector_sum_float_opt

all_vector_sum_int: vector_sum_int vector_sum_int_opt

all_vector_dot: all_vector_dot_float all_vector_dot_int

all_vector_dot_float: vector_dot_float_1 vector_dot_float_2 vector_dot_float_3

all_vector_dot_int: vector_dot_int_1 vector_dot_int_2 vector_dot_int_3

vector_sum_float: $(VECTOR_DIR)/sum/vector_sum_float.cu
	$(CC) $(CFLAGS) -o $(BIN)/$@ $^

vector_sum_float_opt: $(VECTOR_DIR)/sum/vector_sum_float_opt.cu
	$(CC) $(CFLAGS) -o $(BIN)/$@ $^

vector_sum_int: $(VECTOR_DIR)/sum/vector_sum_int.cu
	$(CC) $(CFLAGS) -o $(BIN)/$@ $^

vector_sum_int_opt: $(VECTOR_DIR)/sum/vector_sum_int_opt.cu
	$(CC) $(CFLAGS) -o $(BIN)/$@ $^

vector_dot_float_1: $(VECTOR_DIR)/dot/vector_dot_float_1.cu
	$(CC) $(CFLAGS) -o $(BIN)/$@ $^

vector_dot_float_2: $(VECTOR_DIR)/dot/vector_dot_float_2.cu
	$(CC) $(CFLAGS) -o $(BIN)/$@ $^

vector_dot_float_3: $(VECTOR_DIR)/dot/vector_dot_float_3.cu
	$(CC) $(CFLAGS) -o $(BIN)/$@ $^

vector_dot_int_1: $(VECTOR_DIR)/dot/vector_dot_int_1.cu
	$(CC) $(CFLAGS) -o $(BIN)/$@ $^

vector_dot_int_2: $(VECTOR_DIR)/dot/vector_dot_int_2.cu
	$(CC) $(CFLAGS) -o $(BIN)/$@ $^

vector_dot_int_3: $(VECTOR_DIR)/dot/vector_dot_int_3.cu
	$(CC) $(CFLAGS) -o $(BIN)/$@ $^

##
# basic/rand
##
rand: $(RAND_DIR)/curand.cu
	$(CC) $(CFLAGS) -o $(BIN)/$@ $^

##
# clean
##
