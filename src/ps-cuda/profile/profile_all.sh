#!/bin/bash

##
# Profile application when varying:
#   * block dimension: 32, 64, 128, 256, 512, 1024
#   * optimization level: 0
#   * input dimension: 2048, 4096, 8192, 16384, 32768, 65536
#
# Registers up to 10 experiment replications for each setting.
#
# Profiling results are stored as csv in 'out/profile-[PRECISION]-[BLOCK_DIM]-[OPT_LEVEL].csv'.
##

PS_CUDA_HOME="$(realpath $( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/..)"

PS_CUDA_CMD=${PS_CUDA_HOME}/bin/ps_cuda

OUT_DIR=${PS_CUDA_HOME}/out/profile

NVPROF_OPTS="--csv --log-file"

OUT_DIR=${PS_CUDA_HOME}/out/profile

OUT_FILE_TIME=${OUT_DIR}/timing.csv

ALL_BLOCK_DIM=( 32 64 128 256 512 1024)
ALL_INPUT_DIM=( 2048 4096 8192 16384 32768 65536 )
ALL_OPT_LEVEL=( 0 )
REPLICATIONS=10

if [ ! -d "$OUT_DIR" ]; then
  mkdir -p ${OUT_DIR}
fi

echo "input_dim,block_dim,opt_level,time" > ${OUT_FILE_TIME}

for block_dim in "${ALL_BLOCK_DIM[@]}"; do
  for opt_level in "${ALL_OPT_LEVEL[@]}"; do

    echo "[profiling]> compiling ..."
    make clean all -C ${PS_CUDA_HOME} BLOCK_DIM=BLOCK_DIM_${block_dim} OPT_LEVEL=${opt_level} VERBOSITY=VERBOSITY_OFF PROFILING=PROFILE_KERNEL_OFF CORRECTNESS=CHECK_CORRECTNESS_OFF PRINT=PRINT_OFF

    for input_dim in "${ALL_INPUT_DIM[@]}"; do

      echo "[profiling]> INPUT_DIM: ${input_dim} | BLOCK_DIM: ${block_dim} "

      for replication in $(seq 1 ${REPLICATIONS}); do
        echo "[profiling]> performance evaluation (${replication}/${REPLICATIONS})"
        start_time=$(date +%s%N | cut -b1-13)
        ${PS_CUDA_CMD} ${input_dim}
        stop_time=$(date +%s%N | cut -b1-13)
        elapsed_time=$(( ${stop_time} - ${start_time} ))
        echo -e "${input_dim},${block_dim},${opt_level},${elapsed_time}" >> ${OUT_FILE_TIME}
      done
      echo "-------------------------------------------------------------------"
    done
  done
done
