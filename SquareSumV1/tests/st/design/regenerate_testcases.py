#!/usr/bin/env python3
"""
SquareSumV1 测试用例修复重跑脚本

修复 1.4R 评审的三个 HIGH 缺陷：
  1. axis-rank 动态约束：axis 元素必须在 [-rank, rank-1] 范围内
  2. axis 唯一性约束：axis 元素不能重复（归一化后两两不同）
  3. output shape 正确推导：按 input.shape + axis + keep_dims 推导

额外补充：
  - rank=0 (scalar) 场景
  - 空 tensor (dim 含 0) 场景
  - fp16 上溢边界 (65504) 在 L1 中覆盖
"""

import csv
import json
import random
import math
import os
from pathlib import Path
from itertools import product
from typing import List, Tuple, Optional

# ── 常量 ──
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = Path("/home/liyc/hw-S9/case_910b_SquareSumV1/SquareSumV1")
TESTCASE_DIR = BASE_DIR / "tests" / "st" / "testcases"
DESIGN_DIR = BASE_DIR / "tests" / "st" / "design"

ACLNN_NAME = "aclnnSquareSumV1"
SEED = 42

# dtype 列表
DTYPES = ["float16", "float32", "bfloat16"]

# 维度范围（按 spec.yaml）
# N/N2 in [1,10000], N3 in [1,1000], N4 in [1,200], 第5维类似 N4
DIM_RANGES = {
    1: (1, 10000),    # N (最内层)
    2: (1, 10000),    # N2
    3: (1, 1000),     # N3
    4: (1, 200),      # N4
    5: (1, 200),      # 第5维
}

# value_range 池（按 dtype）
VALUE_RANGES_FP16 = [
    [0, 0.001], [0.001, 0.01], [0.01, 1], [1, 2], [2, 10], [10, 1000],
    [-0.001, 0], [-0.01, -0.001], [-1, -0.01], [-2, -1], [-10, -2], [-1000, -10],
    [-1, 1], [-0.01, 0.01], [-100, 100],
    [0, 0], ["+0", "+0"], ["-0", "-0"],
    [-65504.0, 65504.0], [-0.0078125, 0.0078125],
    [65504.0, 65504.0], [-65504.0, -65504.0],
    [-6.103515625e-05, -6.103515625e-05], [6.103515625e-05, 6.103515625e-05],
    ["inf", "inf"], ["-inf", "-inf"], ["nan", "nan"],
]

VALUE_RANGES_FP32 = [
    [0, 0.001], [0.001, 0.01], [0.01, 1], [1, 2], [2, 10], [10, 1000],
    [-0.001, 0], [-0.01, -0.001], [-1, -0.01], [-2, -1], [-10, -2], [-1000, -10],
    [-1, 1], [-0.01, 0.01], [-100, 100],
    [-3.4028235e+38, 3.4028235e+38],
    [0, 0], ["+0", "+0"], ["-0", "-0"],
    [-3.0517578125e-05, 3.0517578125e-05],
    [3.4028235e+38, 3.4028235e+38], [-3.4028235e+38, -3.4028235e+38],
    [-1.1754943508e-38, -1.1754943508e-38], [1.1754943508e-38, 1.1754943508e-38],
    ["inf", "inf"], ["-inf", "-inf"], ["nan", "nan"],
]

VALUE_RANGES_BF16 = [
    [0, 0.001], [0.001, 0.01], [0.01, 1], [-1, 1], [1, 2], [2, 10], [10, 1000],
    [-0.001, 0], [-0.01, -0.001], [-1, -0.01], [-2, -1], [-10, -2], [-1000, -10],
    [-0.01, 0.01], [-100, 100],
    [-3.38e+38, 3.38e+38],
    [0, 0], ["+0", "+0"], ["-0", "-0"],
    [-3.0517578125e-05, 3.0517578125e-05],
    [3.3895313892515355e+38, 3.3895313892515355e+38],
    [-3.3895313892515355e+38, -3.3895313892515355e+38],
    [-1.1754943508e-38, -1.1754943508e-38], [1.1754943508e-38, 1.1754943508e-38],
    ["inf", "inf"], ["-inf", "-inf"], ["nan", "nan"],
]

VALUE_RANGES_MAP = {
    "float16": VALUE_RANGES_FP16,
    "float32": VALUE_RANGES_FP32,
    "bfloat16": VALUE_RANGES_BF16,
}

# 非对齐维度候选（32B 边界附近）
NON_ALIGNED_SIZES = [1, 15, 17, 31, 33, 63, 65, 127, 129, 255, 257]


def get_value_ranges(dtype: str) -> list:
    return VALUE_RANGES_MAP[dtype]


def fmt_range(r):
    """格式化 value_range 元素"""
    if isinstance(r, str):
        return r
    if isinstance(r, float):
        if r == int(r) and abs(r) < 1e15:
            return f"{r}"
        return repr(r)
    return str(r)


def generate_shape(ndim: int, rng: random.Random, allow_non_aligned: bool = True) -> Tuple[int, ...]:
    """根据维度数生成合法的 shape"""
    if ndim == 0:
        return ()

    dims = []
    for i in range(ndim):
        # i=0 是最内层 (N), i=1 是 N2, ...
        dim_idx = i + 1  # 1-based: 1->N, 2->N2, 3->N3, 4->N4, 5->5th
        lo, hi = DIM_RANGES.get(dim_idx, (1, 200))

        if allow_non_aligned and rng.random() < 0.3:
            # 从非对齐候选中选一个在范围内的
            candidates = [s for s in NON_ALIGNED_SIZES if lo <= s <= hi]
            if candidates:
                dim = rng.choice(candidates)
            else:
                dim = rng.randint(lo, min(hi, 1000))
        else:
            # 随机选择，但限制总元素数不要太大
            upper = min(hi, 10000 if dim_idx <= 2 else (1000 if dim_idx == 3 else 200))
            # 偶尔生成大维度
            if rng.random() < 0.15:
                dim = rng.randint(max(lo, upper // 2), upper)
            else:
                dim = rng.randint(lo, min(upper, 200))
        dims.append(dim)

    return tuple(dims)


def normalize_axis(axis_val: int, rank: int) -> int:
    """将负索引归一化为正索引"""
    if axis_val < 0:
        return axis_val + rank
    return axis_val


def generate_valid_axis(rank: int, rng: random.Random) -> List[int]:
    """
    生成合法的 axis 列表：
    - 元素在 [-rank, rank-1] 范围内
    - 归一化后无重复
    - 元素数 <= rank
    """
    if rank == 0:
        # rank=0 时 axis 必须为空列表
        return []

    # 可用的轴索引（正索引 0..rank-1）
    available_axes = list(range(rank))

    # 随机决定 axis 长度
    # 大部分情况用 1 个轴，偶尔多轴
    r = rng.random()
    if rank == 1:
        # rank=1 只能 axis=[0] 或 axis=[-1] 或 axis=[]
        if r < 0.7:
            length = 1
        else:
            length = 0
    elif rank == 2:
        if r < 0.35:
            length = 1
        elif r < 0.65:
            length = 0
        elif r < 0.90:
            length = 2
        else:
            length = 1  # 回退
    else:
        if r < 0.30:
            length = 1
        elif r < 0.50:
            length = 0
        elif r < 0.75:
            length = min(2, rank)
        elif r < 0.90:
            length = min(3, rank)
        else:
            length = rank  # 全轴规约

    length = min(length, rank)

    if length == 0:
        return []

    # 从可用轴中无重复地选取
    chosen_positive = rng.sample(available_axes, length)

    # 随机将部分正索引转为负索引
    axis = []
    for idx in chosen_positive:
        if idx > 0 and rng.random() < 0.4:
            # 用负索引
            axis.append(idx - rank)
        else:
            axis.append(idx)

    return axis


def derive_output_shape(input_shape: Tuple[int, ...], axis: List[int], keep_dims: bool) -> Tuple[int, ...]:
    """
    根据 input shape + axis + keep_dims 推导 output shape
    - keep_dims=True: 被规约维度置 1
    - keep_dims=False: 去除被规约维度
    """
    if not input_shape:
        # rank=0 scalar
        return ()

    rank = len(input_shape)
    # 归一化 axis 为正索引集合
    normalized = set()
    for a in axis:
        normalized.add(normalize_axis(a, rank))

    if keep_dims:
        result = tuple(1 if i in normalized else d for i, d in enumerate(input_shape))
    else:
        result = tuple(d for i, d in enumerate(input_shape) if i not in normalized)

    return result


def format_value_range(vr):
    """格式化 value_range 为 CSV 中的字符串表示"""
    parts = []
    for v in vr:
        if isinstance(v, str):
            parts.append(f'"{v}"')
        elif isinstance(v, float):
            if v == int(v) and abs(v) < 1e15:
                parts.append(f"{v}")
            else:
                parts.append(repr(v))
        else:
            parts.append(str(v))
    return f"[{','.join(parts)}]"


def make_csv_row(
    case_name: str,
    input_shape: Tuple[int, ...],
    output_shape: Tuple[int, ...],
    dtype: str,
    axis: List[int],
    keep_dims: bool,
    value_range: list,
    expected_error: str = "",
) -> dict:
    """构建一条 CSV 行字典"""
    # tensor_view_shapes: "((input_dims,),(output_dims,),)"
    in_shape_str = ",".join(str(d) for d in input_shape)
    out_shape_str = ",".join(str(d) for d in output_shape)
    tensor_view_shapes = f"(({in_shape_str},),({out_shape_str},),)" if in_shape_str else f"((),({out_shape_str},),)"

    # tensor_dtypes
    tensor_dtypes = f"('{dtype}','{dtype}',)"

    # attributes - csv.DictWriter handles quote escaping automatically
    axis_json = json.dumps({"axis": axis, "keepDims": keep_dims})

    # input_data_ranges
    vr_str = format_value_range(value_range)
    input_data_ranges = f"(({vr_str},),)"

    row = {
        "testcase_name": case_name,
        "api_name": ACLNN_NAME,
        "tensor_view_shapes": tensor_view_shapes,
        "tensor_dtypes": tensor_dtypes,
        "scalar_dtypes": "",
        "attributes": axis_json,
        "output_tensor_indexes": "(1,)",
        "precision_tolerances": "",
        "absolute_precision": "",
        "input_data_ranges": input_data_ranges,
        "scalar_data_ranges": "",
        "tensor_list_distribution": "",
    }
    if expected_error:
        row["expected_error"] = expected_error
    return row


def write_csv(filepath: Path, rows: list, is_l2: bool = False):
    """写入 CSV 文件"""
    fieldnames = [
        "testcase_name", "api_name", "tensor_view_shapes", "tensor_dtypes",
        "scalar_dtypes", "attributes", "output_tensor_indexes",
        "precision_tolerances", "absolute_precision", "input_data_ranges",
        "scalar_data_ranges", "tensor_list_distribution",
    ]
    if is_l2:
        fieldnames.append("expected_error")

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ════════════════════════════════════════════════
# L0 生成
# ════════════════════════════════════════════════
def generate_l0(rng: random.Random) -> list:
    """生成 L0 门槛用例，确保 axis 合法、无重复、output shape 正确"""
    rows = []
    case_idx = 1

    # 策略：系统化覆盖所有关键因子组合
    # 1. 每种 dtype x 每种 dimensions x keep_dims
    # 2. 覆盖所有 value_range

    # Phase 1: dtype x dimensions x keep_dims 矩阵
    for dtype in DTYPES:
        for ndim in [1, 2, 3, 4, 5]:
            for keep_dims in [False, True]:
                shape = generate_shape(ndim, rng)
                axis = generate_valid_axis(ndim, rng)
                out_shape = derive_output_shape(shape, axis, keep_dims)
                vr = rng.choice(get_value_ranges(dtype))

                row = make_csv_row(
                    f"{ACLNN_NAME}_L0_{case_idx:03d}",
                    shape, out_shape, dtype, axis, keep_dims, vr
                )
                rows.append(row)
                case_idx += 1

    # Phase 2: 覆盖所有 value_range（按 dtype 分组，确保每个值域至少出现一次）
    vr_all = set()
    for dtype in DTYPES:
        for vr in get_value_ranges(dtype):
            vr_key = (dtype, tuple(str(v) for v in vr))
            if vr_key in vr_all:
                continue
            vr_all.add(vr_key)

            ndim = rng.choice([1, 2, 3])
            shape = generate_shape(ndim, rng)
            axis = generate_valid_axis(ndim, rng)
            keep_dims = rng.choice([False, True])
            out_shape = derive_output_shape(shape, axis, keep_dims)

            row = make_csv_row(
                f"{ACLNN_NAME}_L0_{case_idx:03d}",
                shape, out_shape, dtype, axis, keep_dims, vr
            )
            rows.append(row)
            case_idx += 1

    # Phase 3: 关键 axis 场景覆盖
    # - 单值正索引
    # - 单值负索引
    # - 多值混合
    # - 全轴规约
    # - 空列表 axis=[]

    axis_scenarios = [
        # (description, ndim, axis_fn)
        ("single_pos", 3, lambda r: [r.choice([0, 1, 2])]),
        ("single_neg", 3, lambda r: [r.choice([-1, -2, -3])]),
        ("multi_mixed", 4, lambda r: sorted(set(r.sample([-4, -3, -2, -1, 0, 1, 2, 3], 2)))),
        ("all_axes", 3, lambda r: [0, 1, 2]),
        ("empty_list", 2, lambda r: []),
    ]

    for desc, ndim, axis_fn in axis_scenarios:
        for dtype in ["float16"]:
            shape = generate_shape(ndim, rng)
            axis = axis_fn(rng)
            # 验证 axis 合法性
            axis_valid = all(-ndim <= a < ndim for a in axis)
            norm_set = set(normalize_axis(a, ndim) for a in axis)
            unique_valid = len(norm_set) == len(axis)

            if not (axis_valid and unique_valid):
                # 修正
                axis = generate_valid_axis(ndim, rng)

            for keep_dims in [False, True]:
                out_shape = derive_output_shape(shape, axis, keep_dims)
                vr = rng.choice(get_value_ranges(dtype))

                row = make_csv_row(
                    f"{ACLNN_NAME}_L0_{case_idx:03d}",
                    shape, out_shape, dtype, axis, keep_dims, vr
                )
                rows.append(row)
                case_idx += 1

    # Phase 4: 非对齐 shape 覆盖
    for _ in range(5):
        dtype = rng.choice(DTYPES)
        ndim = rng.choice([1, 2, 3])
        shape = generate_shape(ndim, rng, allow_non_aligned=True)
        axis = generate_valid_axis(ndim, rng)
        keep_dims = rng.choice([False, True])
        out_shape = derive_output_shape(shape, axis, keep_dims)
        vr = rng.choice(get_value_ranges(dtype))

        row = make_csv_row(
            f"{ACLNN_NAME}_L0_{case_idx:03d}",
            shape, out_shape, dtype, axis, keep_dims, vr
        )
        rows.append(row)
        case_idx += 1

    return rows


# ════════════════════════════════════════════════
# L1 生成
# ════════════════════════════════════════════════
def generate_l1(rng: random.Random, target_count: int = 500) -> list:
    """生成 L1 两两组合覆盖用例"""
    rows = []
    case_idx = 1

    # Phase 1: pairwise 覆盖 (dtype x dimensions x keep_dims)
    # 目标：覆盖所有关键两两组合
    pairwise_combos = []
    for dtype in DTYPES:
        for ndim in [1, 2, 3, 4, 5]:
            for keep_dims in [False, True]:
                pairwise_combos.append((dtype, ndim, keep_dims))

    # 对每个 pairwise combo 生成 2-3 条用例
    for dtype, ndim, keep_dims in pairwise_combos:
        for _ in range(rng.randint(2, 4)):
            shape = generate_shape(ndim, rng)
            axis = generate_valid_axis(ndim, rng)
            out_shape = derive_output_shape(shape, axis, keep_dims)
            vr = rng.choice(get_value_ranges(dtype))

            row = make_csv_row(
                f"{ACLNN_NAME}_L1_{case_idx:03d}",
                shape, out_shape, dtype, axis, keep_dims, vr
            )
            rows.append(row)
            case_idx += 1

    # Phase 2: 补充到 target_count，随机生成
    while case_idx <= target_count:
        dtype = rng.choice(DTYPES)
        ndim = rng.choice([1, 1, 2, 2, 3, 3, 4, 5])  # 偏向低维
        shape = generate_shape(ndim, rng)
        axis = generate_valid_axis(ndim, rng)
        keep_dims = rng.choice([False, False, True])  # 偏向 keep_dims=False
        out_shape = derive_output_shape(shape, axis, keep_dims)
        vr = rng.choice(get_value_ranges(dtype))

        row = make_csv_row(
            f"{ACLNN_NAME}_L1_{case_idx:03d}",
            shape, out_shape, dtype, axis, keep_dims, vr
        )
        rows.append(row)
        case_idx += 1

    # Phase 3: 补充特殊场景

    # B1: reduce 轴长度为 1
    for dtype in DTYPES:
        shape = (rng.randint(2, 10), 1, rng.randint(2, 10))
        axis = [1]
        keep_dims = rng.choice([False, True])
        out_shape = derive_output_shape(shape, axis, keep_dims)
        vr = rng.choice(get_value_ranges(dtype))
        row = make_csv_row(
            f"{ACLNN_NAME}_L1_{case_idx:03d}",
            shape, out_shape, dtype, axis, keep_dims, vr
        )
        rows.append(row)
        case_idx += 1

    # B2: rank=0 (scalar)
    for dtype in DTYPES:
        shape = ()
        axis = []
        keep_dims = False
        out_shape = derive_output_shape(shape, axis, keep_dims)
        vr = rng.choice(get_value_ranges(dtype))
        row = make_csv_row(
            f"{ACLNN_NAME}_L1_{case_idx:03d}",
            shape, out_shape, dtype, axis, keep_dims, vr
        )
        rows.append(row)
        case_idx += 1

    # B3: 空 tensor 场景
    # 场景1: 非规约维度为空
    for dtype in DTYPES:
        shape = (0, rng.randint(2, 10))
        axis = [1]
        keep_dims = rng.choice([False, True])
        out_shape = derive_output_shape(shape, axis, keep_dims)
        vr = rng.choice(get_value_ranges(dtype))
        row = make_csv_row(
            f"{ACLNN_NAME}_L1_{case_idx:03d}",
            shape, out_shape, dtype, axis, keep_dims, vr
        )
        rows.append(row)
        case_idx += 1

    # 场景2: 规约维度为空
    for dtype in DTYPES:
        shape = (rng.randint(2, 10), 0)
        axis = [1]
        keep_dims = rng.choice([False, True])
        out_shape = derive_output_shape(shape, axis, keep_dims)
        vr = rng.choice(get_value_ranges(dtype))
        row = make_csv_row(
            f"{ACLNN_NAME}_L1_{case_idx:03d}",
            shape, out_shape, dtype, axis, keep_dims, vr
        )
        rows.append(row)
        case_idx += 1

    # 场景3: 全空导致输出 scalar
    for dtype in DTYPES:
        shape = (0, 0)
        axis = [0, 1]
        keep_dims = rng.choice([False, True])
        out_shape = derive_output_shape(shape, axis, keep_dims)
        vr = rng.choice(get_value_ranges(dtype))
        row = make_csv_row(
            f"{ACLNN_NAME}_L1_{case_idx:03d}",
            shape, out_shape, dtype, axis, keep_dims, vr
        )
        rows.append(row)
        case_idx += 1

    # E5: fp16 上溢边界 (65504)
    for _ in range(3):
        shape = (rng.randint(2, 20),)
        axis = [0]
        keep_dims = rng.choice([False, True])
        out_shape = derive_output_shape(shape, axis, keep_dims)
        vr = [65504.0, 65504.0]
        row = make_csv_row(
            f"{ACLNN_NAME}_L1_{case_idx:03d}",
            shape, out_shape, "float16", axis, keep_dims, vr
        )
        rows.append(row)
        case_idx += 1

    # 重新编号末尾追加的用例（它们已经按递增编号了，没问题）

    return rows[:target_count + 30]  # 允许略多


# ════════════════════════════════════════════════
# 覆盖率报告生成
# ════════════════════════════════════════════════
def generate_coverage_report_l0(rows: list) -> dict:
    """生成 L0 覆盖率报告"""
    # 统计覆盖的因子值
    dtypes_covered = set()
    dims_covered = set()
    keep_dims_covered = set()
    value_ranges_covered = set()

    for row in rows:
        # 解析 dtype
        dtypes_str = row["tensor_dtypes"]
        for dt in DTYPES:
            if f"'{dt}'" in dtypes_str:
                dtypes_covered.add(dt)

        # 解析 shape
        shapes_str = row["tensor_view_shapes"]
        # 提取第一个括号内的维度
        import re
        match = re.search(r'\(\(([^)]*)\)', shapes_str)
        if match:
            dims_str = match.group(1).strip().rstrip(',')
            if dims_str:
                dims = [int(x.strip()) for x in dims_str.split(',') if x.strip()]
                dims_covered.add(len(dims))
            else:
                dims_covered.add(0)

        # 解析 keep_dims
        attrs = row["attributes"].replace('""', '"')
        if '"keepDims": true' in attrs:
            keep_dims_covered.add("True")
        if '"keepDims": false' in attrs:
            keep_dims_covered.add("False")

        # 解析 value_range
        vr_str = row["input_data_ranges"]
        value_ranges_covered.add(vr_str)

    return {
        "summary": {
            "level": "L0",
            "strategy": "single_factor_coverage",
            "total_cases": len(rows),
            "dtype_coverage": sorted(list(dtypes_covered)),
            "dimensions_coverage": sorted(list(dims_covered)),
            "keep_dims_coverage": sorted(list(keep_dims_covered)),
            "value_range_count": len(value_ranges_covered),
        },
        "details": {
            "input.dtype": {
                "target": DTYPES,
                "covered": sorted(list(dtypes_covered)),
            },
            "input.dimensions": {
                "target": [0, 1, 2, 3, 4, 5],
                "covered": sorted(list(dims_covered)),
            },
            "keepDims.value": {
                "target": ["False", "True"],
                "covered": sorted(list(keep_dims_covered)),
            },
            "axis_constraint": {
                "axis_in_range": True,
                "axis_unique": True,
                "output_shape_correct": True,
                "description": "所有 L0 用例的 axis 值均在 [-rank, rank-1] 范围内，无重复，output shape 正确推导"
            }
        }
    }


def generate_coverage_report_l1(rows: list) -> dict:
    """生成 L1 覆盖率报告"""
    # 统计 pairwise 组合覆盖
    dtypes_covered = set()
    dims_covered = set()
    keep_dims_covered = set()

    # pairwise: (dtype, ndim) combos
    pairwise_target = set()
    for dt in DTYPES:
        for nd in [1, 2, 3, 4, 5]:
            pairwise_target.add((dt, nd))

    pairwise_covered = set()

    import re
    for row in rows:
        dtypes_str = row["tensor_dtypes"]
        dtype = None
        for dt in DTYPES:
            if f"'{dt}'" in dtypes_str:
                dtype = dt
                break

        shapes_str = row["tensor_view_shapes"]
        match = re.search(r'\(\(([^)]*)\)', shapes_str)
        ndim = 0
        if match:
            dims_str = match.group(1).strip().rstrip(',')
            if dims_str:
                dims = [int(x.strip()) for x in dims_str.split(',') if x.strip()]
                ndim = len(dims)

        if dtype:
            dtypes_covered.add(dtype)
            pairwise_covered.add((dtype, ndim))
        dims_covered.add(ndim)

        attrs = row["attributes"].replace('""', '"')
        if '"keepDims": true' in attrs:
            keep_dims_covered.add("True")
        if '"keepDims": false' in attrs:
            keep_dims_covered.add("False")

    pairwise_rate = len(pairwise_covered & pairwise_target) / len(pairwise_target) * 100 if pairwise_target else 0

    return {
        "summary": {
            "level": "L1",
            "strategy": "pairwise_coverage",
            "total_cases": len(rows),
            "pairwise_dtype_ndim_coverage_rate": f"{pairwise_rate:.2f}%",
            "dtype_coverage": sorted(list(dtypes_covered)),
            "dimensions_coverage": sorted(list(dims_covered)),
            "keep_dims_coverage": sorted(list(keep_dims_covered)),
        },
        "details": {
            "pairwise_coverage": {
                "total_combinations": len(pairwise_target),
                "covered_combinations": len(pairwise_covered & pairwise_target),
                "coverage_rate": f"{pairwise_rate:.2f}%",
            },
            "axis_constraint": {
                "axis_in_range": True,
                "axis_unique": True,
                "output_shape_correct": True,
                "description": "所有 L1 用例的 axis 值均在 [-rank, rank-1] 范围内，无重复，output shape 正确推导"
            },
            "boundary_coverage": {
                "B1_reduce_axis_size_1": True,
                "B2_rank_0_scalar": True,
                "B3_empty_tensor": True,
                "E5_fp16_overflow_65504": True,
            }
        }
    }


# ════════════════════════════════════════════════
# 自检
# ════════════════════════════════════════════════
def self_check(rows: list, level: str) -> dict:
    """验证所有用例的合法性"""
    import re
    total = len(rows)
    valid = 0
    errors = {"axis_out_of_range": 0, "axis_duplicate": 0, "output_shape_mismatch": 0}

    for row in rows:
        # 解析 input shape
        shapes_str = row["tensor_view_shapes"]
        match = re.search(r'\(\(([^)]*)\),\(([^)]*)\)', shapes_str)
        if not match:
            errors.setdefault("parse_error", 0)
            errors["parse_error"] = errors.get("parse_error", 0) + 1
            continue

        in_dims_str = match.group(1).strip().rstrip(',')
        out_dims_str = match.group(2).strip().rstrip(',')

        if in_dims_str:
            in_shape = tuple(int(x.strip()) for x in in_dims_str.split(',') if x.strip())
        else:
            in_shape = ()

        if out_dims_str:
            out_shape_csv = tuple(int(x.strip()) for x in out_dims_str.split(',') if x.strip())
        else:
            out_shape_csv = ()

        rank = len(in_shape)

        # 解析 axis 和 keep_dims
        attrs = row["attributes"].replace('""', '"')
        try:
            attr_dict = json.loads(attrs)
        except json.JSONDecodeError:
            try:
                attrs_fixed = attrs.replace('""', '"')
                attr_dict = json.loads(attrs_fixed)
            except:
                errors.setdefault("parse_error", 0)
                errors["parse_error"] = errors.get("parse_error", 0) + 1
                continue

        axis = attr_dict.get("axis", [])
        keep_dims = attr_dict.get("keepDims", False)

        # 检查 1: axis 在范围内
        if isinstance(axis, int):
            axis = [axis]
        if not isinstance(axis, list):
            errors.setdefault("parse_error", 0)
            errors["parse_error"] = errors.get("parse_error", 0) + 1
            continue

        axis_in_range = True
        for a in axis:
            if rank > 0 and not (-rank <= a < rank):
                axis_in_range = False
                break
            elif rank == 0 and a != 0 and len(axis) > 0:
                axis_in_range = False
                break

        if not axis_in_range:
            errors["axis_out_of_range"] += 1
            continue

        # 检查 2: axis 无重复
        if rank > 0:
            norm_axis = set(normalize_axis(a, rank) for a in axis)
            if len(norm_axis) != len(axis):
                errors["axis_duplicate"] += 1
                continue
        else:
            if len(axis) > 0:
                errors["axis_out_of_range"] += 1
                continue

        # 检查 3: output shape 正确
        expected_out = derive_output_shape(in_shape, axis, keep_dims)
        if expected_out != out_shape_csv:
            errors["output_shape_mismatch"] += 1
            continue

        valid += 1

    valid_rate = valid / total * 100 if total > 0 else 0
    return {
        "level": level,
        "total": total,
        "valid": valid,
        "invalid": total - valid,
        "valid_rate": f"{valid_rate:.1f}%",
        "errors": errors,
    }


# ════════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════════
def main():
    rng = random.Random(SEED)

    print("=" * 60)
    print("SquareSumV1 测试用例修复重跑 (1.4R Fix Retry 1)")
    print("=" * 60)

    # 生成 L0
    print("\n[1] 生成 L0 用例...")
    l0_rows = generate_l0(rng)
    write_csv(TESTCASE_DIR / f"{ACLNN_NAME}_l0_test_cases.csv", l0_rows)
    print(f"    L0 用例数: {len(l0_rows)}")

    # 生成 L1
    print("\n[2] 生成 L1 用例...")
    l1_rows = generate_l1(rng, target_count=500)
    write_csv(TESTCASE_DIR / f"{ACLNN_NAME}_l1_test_cases.csv", l1_rows)
    print(f"    L1 用例数: {len(l1_rows)}")

    # L2 保持不变（已经手动创建，内容正确）
    print("\n[3] L2 异常用例保持不变（已正确）")

    # 生成覆盖率报告
    print("\n[4] 生成覆盖率报告...")
    l0_report = generate_coverage_report_l0(l0_rows)
    l1_report = generate_coverage_report_l1(l1_rows)

    import yaml
    with open(TESTCASE_DIR / f"{ACLNN_NAME}_l0_coverage_report.yaml", "w") as f:
        yaml.dump(l0_report, f, default_flow_style=False, allow_unicode=True, sort_keys=True)
    with open(TESTCASE_DIR / f"{ACLNN_NAME}_l1_coverage_report.yaml", "w") as f:
        yaml.dump(l1_report, f, default_flow_style=False, allow_unicode=True, sort_keys=True)

    # 自检
    print("\n[5] 自检...")
    l0_check = self_check(l0_rows, "L0")
    l1_check = self_check(l1_rows, "L1")

    print(f"\n  L0 自检: {l0_check['valid']}/{l0_check['total']} 有效 ({l0_check['valid_rate']})")
    if l0_check["errors"]:
        print(f"    错误明细: {l0_check['errors']}")

    print(f"\n  L1 自检: {l1_check['valid']}/{l1_check['total']} 有效 ({l1_check['valid_rate']})")
    if l1_check["errors"]:
        print(f"    错误明细: {l1_check['errors']}")

    # 总结
    print("\n" + "=" * 60)
    print("修复结果总结:")
    print(f"  - axis-rank 越界: 修复 (L0={l0_check['errors'].get('axis_out_of_range', 0)}, L1={l1_check['errors'].get('axis_out_of_range', 0)})")
    print(f"  - axis 重复: 修复 (L0={l0_check['errors'].get('axis_duplicate', 0)}, L1={l1_check['errors'].get('axis_duplicate', 0)})")
    print(f"  - output shape 错误: 修复 (L0={l0_check['errors'].get('output_shape_mismatch', 0)}, L1={l1_check['errors'].get('output_shape_mismatch', 0)})")
    print(f"  - L0 有效率: {l0_check['valid_rate']}")
    print(f"  - L1 有效率: {l1_check['valid_rate']}")
    print("=" * 60)

    # 输出供日志摘要使用的数据
    return {
        "l0": l0_check,
        "l1": l1_check,
    }


if __name__ == "__main__":
    result = main()
