#!/bin/bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 <vendor_root> <new_output_dir>" >&2
    exit 2
fi

VENDOR_ROOT="$1"
OUTPUT_DIR="$2"
PROJECT_ROOT="/home/liyc/hw-S9/case_910b_SquareSumV1"
PYTHON_BIN="/home/ma-user/anaconda3/envs/MindSpore/bin/python"
PROFILE_DRIVER="${SQUARESUMV1_PROFILE_DRIVER:-${PROJECT_ROOT}/SquareSumV1/npu_acceptance_perf_batch_driver.py}"

if [[ -e "${OUTPUT_DIR}" ]]; then
    echo "output directory already exists: ${OUTPUT_DIR}" >&2
    exit 2
fi
test -f "${VENDOR_ROOT}/bin/set_env.bash"
test -x "${PYTHON_BIN}"
test -f "${PROFILE_DRIVER}"

set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source "${VENDOR_ROOT}/bin/set_env.bash"
set -u
export SQUARESUMV1_OPP_ROOT="${VENDOR_ROOT}"
export ASCEND_RT_VISIBLE_DEVICES=0

mkdir -p "${OUTPUT_DIR}"
msprof \
    --output="${OUTPUT_DIR}" \
    --ai-core=on \
    --aic-mode=task-based \
    --aic-metrics=PipeUtilization \
    --task-time=on \
    --ascendcl=on \
    --application="${PYTHON_BIN} ${PROFILE_DRIVER}" \
    > "${OUTPUT_DIR}/msprof.log" 2>&1
