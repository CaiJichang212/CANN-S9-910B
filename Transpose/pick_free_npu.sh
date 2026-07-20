#!/bin/bash
# pick_free_npu.sh —— 共享 8 卡服务器上自动检测并选定一张空闲 NPU 卡。
#
# 用法（必须用 source 调用，以便环境变量传给当前 shell 的后续命令）：
#     source ./pick_free_npu.sh
#
# 行为：
#   1. 若已显式设 ASCEND_RT_VISIBLE_DEVICES，则尊重沿用，不覆盖。
#   2. 否则解析 `npu-smi info`，取第一张 "No running processes" 的卡，
#      export ASCEND_RT_VISIBLE_DEVICES=<id>。
#   3. 全部被占时打印明确错误并 return 1（绝不默认抢卡 0，避免影响他人）。
#
# 判定「空闲」的依据：npu-smi 底部 process 表显示 "No running processes found in NPU X"。
# HBM 基线占用 ~3.4GB 属驱动开销，不代表被占。

pick_free_npu() {
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    # 1) 用户已显式指定 -> 沿用
    if [ -n "${ASCEND_RT_VISIBLE_DEVICES:-}" ]; then
        echo "[pick_free_npu] 已设 ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES，沿用" >&2
        return 0
    fi

    # 2) 解析 npu-smi info，找空闲卡
    local npu_out
    npu_out="$(npu-smi info 2>/dev/null)"
    if [ -z "$npu_out" ]; then
        echo "[pick_free_npu] 错误：npu-smi info 无输出，驱动可能异常" >&2
        return 1
    fi

    # 匹配 "No running processes found in NPU <id>"，提取 id。
    # 共享服务器上低编号卡(0,1,2,3)最容易被「不设环境变量的用户」默认抢占，
    # 故优先选高编号空闲卡（取最大 id），降低与他人碰撞概率。
    local free_cards
    free_cards="$(echo "$npu_out" | grep -oE "No running processes found in NPU [0-9]+" | grep -oE "[0-9]+$" | sort -n)"
    if [ -z "$free_cards" ]; then
        echo "[pick_free_npu] 错误：8 张卡当前全部被占用，请稍后重试或手动指定：" >&2
        echo "    export ASCEND_RT_VISIBLE_DEVICES=<空闲卡id>  # 先 npu-smi info 查看" >&2
        return 1
    fi

    local picked
    picked="$(echo "$free_cards" | tail -1)"
    export ASCEND_RT_VISIBLE_DEVICES="$picked"
    echo "[pick_free_npu] 选定空闲卡 NPU $picked（空闲卡列表: $(echo "$free_cards" | tr '\n' ' '))" >&2
    return 0
}

# source 时自动执行；直接 bash 运行时只定义函数不执行（便于测试）。
if [ "${BASH_SOURCE[0]}" = "${0:-}" ]; then
    # 直接执行模式：打印建议
    echo "本脚本应被 source 调用：source ./pick_free_npu.sh" >&2
    pick_free_npu
else
    pick_free_npu
fi
