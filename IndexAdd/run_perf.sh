#!/bin/bash
# IndexAdd 性能采集驱动：逐 case × {custom, builtin} msprof 采集 + 解析
# 用法: PERF_DEVICE=2 bash run_perf.sh [case_id ...]   (默认全部 20 case)
#
# custom  — vendors/customize 正常 + LD 含 customize → 自定义 aclnnIndexAdd (走 custom_op)
# builtin — 移走 vendors/customize → dlopen 失败回退 libopapi.so 内置 aclnnIndexAdd
#           (torch_npu 也优先加载 libcust_opapi.so，故必须整体移走 vendors/customize)
set -uo pipefail
cd "$(dirname "$0")"

DEV=${PERF_DEVICE:-2}
VEND=${ASCEND_OPP_PATH}/vendors/customize
OUT=${PERF_OUT:-./perf_out}
RESULTS=$OUT/results.jsonl
mkdir -p "$OUT"
: > "$RESULTS"

CASES=("$@")
if [ ${#CASES[@]} -eq 0 ]; then
    CASES=(c01 c02 c03 c04 c05 c06 c07 c08 c09 c10 c11 c12 c13 c14 c15 c16 c17 c18 c19 c20)
fi

restore() {
    if [ -d "${VEND}.bak" ]; then
        mv "${VEND}.bak" "$VEND"
        echo "[restore] vendors/customize 已恢复"
    fi
}
trap restore EXIT

clean_ld() { echo "${LD_LIBRARY_PATH}" | tr ':' '\n' | grep -v "vendors/customize/op_api/lib" | paste -sd:; }

run_one() {  # $1=case_id $2=mode
    local cid=$1 mode=$2
    rm -rf "${OUT}/${cid}_${mode}"
    echo "=== ${cid} / ${mode} ==="
    PERF_DEVICE=$DEV PERF_MODE=$mode timeout 300 msprof \
        --application="python3 perf_run.py ${cid}" \
        --output="${OUT}/${cid}_${mode}" 2>&1 \
        | grep -E "\[CASE\]|\[VERIFY\]|RuntimeError|timed out" || true
    if compgen -G "${OUT}/${cid}_${mode}/PROF_*" > /dev/null; then
        python3 parse_perf.py "${OUT}/${cid}_${mode}" "$cid" "$mode" >> "$RESULTS"
    else
        echo "{\"case_id\":\"${cid}\",\"mode\":\"${mode}\",\"error\":\"no prof dir\"}" >> "$RESULTS"
    fi
}

# ---- Phase 1: custom（vendors/customize 正常）----
export LD_LIBRARY_PATH=${VEND}/op_api/lib:${LD_LIBRARY_PATH}
for cid in "${CASES[@]}"; do run_one "$cid" custom; done

# ---- Phase 2: builtin（移走 vendors/customize → 纯内置）----
mv "$VEND" "${VEND}.bak"
export LD_LIBRARY_PATH="$(clean_ld)"
for cid in "${CASES[@]}"; do run_one "$cid" builtin; done

echo "=== DONE: $RESULTS ==="
