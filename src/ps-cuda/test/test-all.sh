#!/bin/bash

PS_CUDA_HOME="$(realpath $( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/..)"

PS_CUDA_CMD=${PS_CUDA_HOME}/bin/ps_cuda

OUT_DIR=${PS_CUDA_HOME}/out/correctness

ALL_BLOCK_DIM=( 32 64 128 256 512 1024 )
ALL_N_DIM=( 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 )

if [ ! -d "$OUT_DIR" ]; then
  mkdir -p ${OUT_DIR}
fi

for block_dim in "${ALL_BLOCK_DIM[@]}"; do

  echo "[correctness]> compiling ..."
  make clean all -C ${PS_CUDA_HOME} BLOCK_DIM=BLOCK_DIM_${block_dim} VERBOSITY=VERBOSITY_OFF PROFILING=PROFILE_KERNEL_OFF CORRECTNESS=CHECK_CORRECTNESS_ON

  for n_dim in "${ALL_N_DIM[@]}"; do

    OUT_FILE=${OUT_DIR}/correctness-${n_dim}-${block_dim}.out

    if [ ! -e "$OUT_FILE" ] ; then
        touch "$OUT_FILE"
    fi

    echo "[correctness]> N_ELEMENTS: ${n_dim} | BLOCK_DIM: ${block_dim}"

    echo "[correctness]> running ..."
    ${PS_CUDA_CMD} ${n_dim} > ${OUT_FILE}

    if grep -q "CORRECT" "${OUT_FILE}"; then
    	echo "[correctness]> CORRECT"
    else
      echo "[correctness]> ERROR"
    fi

    echo "---------------------------------------------------------------------"
  done
done
