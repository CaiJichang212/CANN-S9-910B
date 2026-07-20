#!/bin/bash
# run_all.sh —— 遍历 test_op.py 所有 case，仅做精度验证（不走 msprof），报告通过/失败。
# 用法: bash run_all.sh
# 前置: 已 bash build.sh && bash build_out/custom_opp_*.run && pip install dist/*.whl
set +e
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export LD_LIBRARY_PATH=$ASCEND_OPP_PATH/vendors/customize/op_api/lib/:$LD_LIBRARY_PATH

# 共享 8 卡服务器：自动选一张空闲 NPU 卡（必须 source，以继承 ASCEND_RT_VISIBLE_DEVICES）
source "$(dirname "$0")/pick_free_npu.sh" || { echo "[run_all] 无空闲 NPU 卡，退出" >&2; exit 1; }

# 确保 whl 已装
if ! python3 -c "import custom_ops_lib" 2>/dev/null; then
    echo "[INFO] custom_ops_lib 未装，安装 dist/*.whl"
    pip3 install dist/custom_ops*.whl --force-reinstall 2>&1 | tail -1
fi

NCASE=12
PASS=0
FAIL=0
FAILED_CASES=""
for i in $(seq 1 $NCASE); do
    echo "==================== case$i ===================="
    OUT=$(timeout 120 python3 test_op.py $i 2>/dev/null)
    echo "$OUT"
    if echo "$OUT" | grep -q "verify result pass"; then
        echo ">>> case$i PASS"
        PASS=$((PASS+1))
    else
        echo ">>> case$i FAIL"
        FAIL=$((FAIL+1))
        FAILED_CASES="$FAILED_CASES $i"
    fi
done
echo ""
echo "==================== SUMMARY ===================="
echo "PASS=$PASS FAIL=$FAIL / total=$NCASE"
[ -n "$FAILED_CASES" ] && echo "FAILED CASES:$FAILED_CASES"
