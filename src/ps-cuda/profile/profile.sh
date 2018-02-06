#!/bin/bash

##
# Profile the currently compiled application.
#
# Profiling results are stored in 'out/profile-[DATE].csv'.
##

PS_CUDA_HOME="$(realpath $( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/..)"

PS_CUDA_CMD=${PS_CUDA_HOME}/bin/ps_cuda

NVPROF_CMD=nvprof

OUT_DIR=${PS_CUDA_HOME}/out/profile

OUT_FILE="${OUT_DIR}/profile-$(date +%Y-%m-%d_%H-%M-%S).csv"

NVPROF_OPTS="--csv --log-file ${OUT_FILE}"

if [ ! -d "$OUT_DIR" ]; then
  mkdir -p ${OUT_DIR}
fi

${NVPROF_CMD} ${NVPROF_OPTS} ${PS_CUDA_CMD}
