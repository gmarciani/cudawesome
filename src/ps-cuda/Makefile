##
# Directories
##
BINDIR = ./bin

SRCDIR = ./src

##
# Variables
##
BLOCK_DIM = BLOCK_DIM_1024
GPU_ARCHITECTURE = sm_35
OPT_LEVEL = 0
VERBOSITY = VERBOSITY_OFF
PROFILING = PROFILE_KERNEL_ON
CORRECTNESS = CHECK_CORRECTNESS_ON
PRINT = PRINT_ON

#--csv
NVPROF_OPTS = --log-file $(shell pwd)/eval/out/profile-$(shell date +%Y-%m-%d_%H-%M-%S).out

##
# CUDA Compilation
##
NVCC = nvcc

NVCC_CCOPTS = --compiler-options -m64,-O$(OPT_LEVEL),-Wall,-Wextra,-Wno-unused-parameter

NVCC_MFLAGS = --define-macro LINUX,$(BLOCK_DIM),$(VERBOSITY),$(PROFILING),$(CORRECTNESS),$(PRINT)

NVCC_IFLAGS = --include-path $(SRCDIR)

NVCC_CFLAGS = --machine 64 --gpu-architecture=$(GPU_ARCHITECTURE) -x cu $(NVCC_IFLAGS) $(NVCC_CCOPTS) --default-stream per-thread

NVCC_LFLAGS = -lm

NVCC_DFLAGS = --debug --device-debug


##
.PHONY: all debug clean makedirs profile profile_all test_all


all: makedirs ps_cuda

debug: NVCC_CFLAGS += $(NVCC_DFLAGS)
debug: all

# clean everything
clean: clean_part
	rm -rf $(BINDIR)

# clean partial results
clean_part:
	rm -rf ./log
	rm -rf ./eval

makedirs:
	mkdir -p $(BINDIR)
	mkdir -p ./log
	mkdir -p ./eval/out
	mkdir -p ./test/out

test_all:
	bash ./test/test-all.sh

profile:
	bash ./profile/profile.sh

profile_all:
	bash ./profile/profile_all.sh

##
# Targets: ps_cuda
##
PS_DEPS = $(SRCDIR)/common/mainutil.c

PS_GPU_DEPS = $(SRCDIR)/gpu/ps.cu

PS_CUDA_DEPS = $(PS_DEPS) $(PS_GPU_DEPS)

ps_cuda: $(SRCDIR)/ps_cuda.cu $(PS_CUDA_DEPS)
	$(NVCC) -o $(BINDIR)/$@ $^ $(NVCC_CFLAGS) $(NVCC_MFLAGS) $(NVCC_LFLAGS)
