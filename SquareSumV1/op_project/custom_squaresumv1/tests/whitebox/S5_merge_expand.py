#!/usr/bin/env python3
"""S5 Merge & Expand — 5c (合并 + 空 tensor 补全) + 5d (data_range 展开)

5c: 过滤 > 1 亿 path case → 补全空 tensor → 合并 path + network + empty → low (all normal)
5d: 基于 low 做 data_range one-hot + 全统一展开 → high
"""

import copy
import json
import math
import os
import re

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

import sys
sys.path.insert(0, _BASE_DIR)
from S5_case_mapper import supplement_empty_cases


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

MAX_NUMEL = 100_000_000  # 1 亿

NON_NORMAL = ["zero", "extreme", "negative", "tiny_pos",
              "all_ones", "near_zero", "with_inf", "with_nan"]

_COMPACT_KEYS = ("shape", "dtype", "id", "_group", "_data_range")


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def set_all_data_range(case, dr):
    """设置所有输入 tensor 的 _data_range。"""
    for spec in case["tensors"]["inputs"].values():
        if spec is None:
            continue
        spec["_data_range"] = dr
    return case


def _tensor_numel(spec):
    """计算 tensor 元素数，None 返回 0。"""
    if spec is None:
        return 0
    shape = spec.get("shape", [])
    return math.prod(shape) if shape else 0


# ---------------------------------------------------------------------------
# 5d: data_range 展开
# ---------------------------------------------------------------------------

def expand_high(cases):
    """one-hot + 全统一 展开。"""
    expanded = []
    for c in cases:
        input_names = [n for n, s in c["tensors"]["inputs"].items() if s is not None]

        # 全 normal
        nc = copy.deepcopy(c)
        nc["id"] = f"{c['id']}_all_normal"
        set_all_data_range(nc, "normal")
        expanded.append(nc)

        # 全统一 non-normal
        for dr in NON_NORMAL:
            nc = copy.deepcopy(c)
            nc["id"] = f"{c['id']}_all_{dr}"
            set_all_data_range(nc, dr)
            expanded.append(nc)

        # one-hot per input
        for inp in input_names:
            for dr in NON_NORMAL:
                nc = copy.deepcopy(c)
                nc["id"] = f"{c['id']}_{inp}_{dr}"
                for name, spec in nc["tensors"]["inputs"].items():
                    if spec is None:
                        continue
                    spec["_data_range"] = dr if name == inp else "normal"
                expanded.append(nc)

    return expanded


# ---------------------------------------------------------------------------
# JSON 格式化
# ---------------------------------------------------------------------------

def _compact_json(text):
    """将 indent=2 的 JSON 中特定 key 的值压缩为单行。"""
    for key in _COMPACT_KEYS:
        # shape: [多行] → [单行]
        text = re.sub(
            rf'("{key}"): \[\s*\n((?:\s+[^\n]+,\n)*\s+[^\n]+\n\s*)\]',
            lambda m: f'{m.group(1)}: [{", ".join(l.strip().rstrip(",") for l in m.group(2).strip().splitlines() if l.strip())}]',
            text, flags=re.MULTILINE
        )
        # scalar values: "value" or number
        text = re.sub(
            rf'("{key}"): ("[^"]*"|\d+(?:\.\d+)?(?:e[+-]?\d+)?)',
            rf'\1: \2',
            text, flags=re.MULTILINE
        )
    return text


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    # 加载
    with open(os.path.join(_BASE_DIR, "S2P1_operator_model.json")) as f:
        sm = json.load(f)["shape_mapping"]

    with open(os.path.join(_BASE_DIR, "S5_mapped_cases_path.json")) as f:
        path_cases = json.load(f)["cases"]

    with open(os.path.join(_BASE_DIR, "S5_mapped_cases_network.json")) as f:
        net_cases = json.load(f)["cases"]

    # 5c: 过滤 path case（元素数 > 1 亿）
    path_filtered = []
    path_dropped = 0
    for c in path_cases:
        ok = all(
            v is None or _tensor_numel(v) <= MAX_NUMEL
            for v in c["tensors"]["inputs"].values()
        )
        if ok:
            path_filtered.append(c)
        else:
            path_dropped += 1
    print(f"5c: filtered path cases: {len(path_cases)} → {len(path_filtered)} "
          f"(dropped {path_dropped} with > {MAX_NUMEL} elements)")

    # 5c: 空 tensor 补全
    empty_cases = supplement_empty_cases(sm, path_filtered)
    print(f"5c: generated {len(empty_cases)} empty tensor variants")
    for ec in empty_cases:
        print(f"     {ec['id']}: group={ec['params'].get('_group', '?')}")

    # 5c: 合并 path + network + empty → low（all normal）
    combined = path_filtered + net_cases + empty_cases
    low = [set_all_data_range(copy.deepcopy(c), "normal") for c in combined]

    low_file = os.path.join(_BASE_DIR, "S5_mapped_cases_low.json")
    with open(low_file, "w") as f:
        raw = json.dumps({"cases": low}, indent=2, ensure_ascii=False)
        f.write(_compact_json(raw))
    print(f"5c: wrote {len(low)} cases to {low_file}")

    # 5d: high — 基于 filtered path + empty + network 做 data_range 展开
    # 注意：empty case 在 high 中固定 normal，不参与 data_range 交叉
    empty_normal = [set_all_data_range(copy.deepcopy(c), "normal") for c in empty_cases]
    high = expand_high(path_filtered) + empty_normal + expand_high(net_cases)

    high_file = os.path.join(_BASE_DIR, "S5_mapped_cases_high.json")
    with open(high_file, "w") as f:
        json.dump({"cases": high}, f, ensure_ascii=False)
    print(f"5d: wrote {len(high)} cases to {high_file}")

    # 统计
    print()
    print("=" * 60)
    print("Summary:")
    print(f"  path (filtered): {len(path_filtered)}")
    print(f"  network:         {len(net_cases)}")
    print(f"  empty:           {len(empty_cases)}")
    print(f"  low (all normal): {len(low)}")
    print(f"  high (expanded):  {len(high)}")
    # high 展开因子
    path_factor = len(path_filtered) * (1 + len(NON_NORMAL) + 1 * len(NON_NORMAL)) if path_filtered else 0
    net_factor = len(net_cases) * (1 + len(NON_NORMAL) + 1 * len(NON_NORMAL)) if net_cases else 0
    print(f"  expected high: {path_factor} + {len(empty_cases)} + {net_factor} = {path_factor + len(empty_cases) + net_factor}")
    print("=" * 60)


if __name__ == "__main__":
    main()
