"""IndexAdd 系统化性能测试用例矩阵。

每个 case 标注诊断标签，用于瓶颈归因：
  - copy-bound / scatter-bound     : self 体积 vs source 体积主导
  - atomic / owned-rmw             : kernel 走哪条 scatter 路径（由 host tiling 决定）
  - aligned / unaligned            : afterDimSize*dtypeSize 是否 32B 对齐
  - cast / no-cast                 : int8/bf16 需 Cast
  - index-unique / index-repeat / index-extreme : index 重复度
  - dim-head / dim-mid / dim-tail  : dim 在第 0/中/末维
  - large-vector / scalar-scatter  : afterDimSize 大（连续段长）vs =1（纯散粒度）

对齐判定（host tiling 的 atomicEnabled 条件）：
  vectorBytes = afterDimSize * dtypeSize
  atomic  <=>  dtype != bf16  AND  vectorBytes >= 256  AND  vectorBytes % 32 == 0
  即: fp32/int32 afterDim%8==0 且 >=64; fp16 afterDim%16==0 且>=128;
      int8 afterDim%32==0 且>=256; bf16 恒走 owner。
"""

import numpy as np

# index 生成模式
UNIQUE = "unique"      # 全不重复
REPEAT = "repeat"      # 允许重复（随机）
EXTREME = "extreme"    # 极端重复（全部指向少数几个输出行）


def _index(spec, dim_len, m, rng):
    if spec == UNIQUE:
        if m <= dim_len:
            idx = rng.choice(dim_len, m, replace=False)
        else:
            idx = rng.randint(0, dim_len, m)
    elif spec == EXTREME:
        # 全部指向 0..K-1，K = max(1, dim_len//64)，模拟热点行被反复累加
        k = max(1, dim_len // 64)
        idx = rng.randint(0, k, m)
    else:  # REPEAT
        idx = rng.randint(0, dim_len, m)
    return idx.astype(np.int32)


# (id, np_dtype, torch_dtype_key, self_shape, dim, M, index_spec, labels)
CASES = [
    # === 第一组：dtype 覆盖（[2000,1024] dim=0 M=2000 repeat，全 dtype 对齐）===
    # afterDim=1024: fp32/fp16/bf16/int32/int8 均 32B 对齐 → 除 bf16 外全走 atomic
    ("c01", np.float32, None,   [2000, 1024], 0, 2000, REPEAT,
     ["dtype-fp32", "atomic", "aligned", "no-cast", "dim-head"]),
    ("c02", np.float16, None,   [2000, 1024], 0, 2000, REPEAT,
     ["dtype-fp16", "atomic", "aligned", "no-cast", "dim-head"]),
    ("c03", np.float32, "bf16", [2000, 1024], 0, 2000, REPEAT,
     ["dtype-bf16", "owned-rmw", "aligned", "cast", "dim-head"]),
    ("c04", np.int32,   None,   [2000, 1024], 0, 2000, REPEAT,
     ["dtype-int32", "atomic", "aligned", "no-cast", "dim-head"]),
    ("c05", np.int8,    None,   [2000, 1024], 0, 2000, REPEAT,
     ["dtype-int8", "atomic", "aligned", "cast", "dim-head"]),

    # === 第二组：对齐 vs 非对齐（fp32 dim=0 M=2000 repeat；c01 为对齐基准）===
    ("c06", np.float32, None,   [2000, 993],  0, 2000, REPEAT,
     ["unaligned", "owned-rmw", "dtype-fp32", "dim-head"]),   # 993*4=3972, %32=4
    ("c07", np.float32, None,   [2000, 997],  0, 2000, REPEAT,
     ["unaligned", "owned-rmw", "dtype-fp32", "dim-head"]),   # 997*4=3988, %32=20

    # === 第三组：index 重复度（fp32 [2000,1024] dim=0 对齐；c01 为 repeat 基准）===
    ("c08", np.float32, None,   [2000, 1024], 0, 2000, UNIQUE,
     ["index-unique", "atomic", "aligned", "dtype-fp32"]),
    ("c09", np.float32, None,   [2000, 1024], 0, 2000, EXTREME,
     ["index-extreme", "atomic", "aligned", "dtype-fp32"]),

    # === 第四组：dim 位置（fp32 [128,2000,64] M=1000 repeat）===
    ("c10", np.float32, None,   [128, 2000, 64], 0, 1000, REPEAT,
     ["dim-head", "large-vector", "atomic", "aligned", "dtype-fp32"]),  # afterDim=128000
    ("c11", np.float32, None,   [128, 2000, 64], 1, 1000, REPEAT,
     ["dim-mid", "atomic", "aligned", "dtype-fp32"]),   # afterDim=64, vectorBytes=256
    ("c12", np.float32, None,   [16, 500, 64], 2, 200, REPEAT,
     ["dim-tail", "scalar-scatter", "owned-rmw", "dtype-fp32"]),  # afterDim=1

    # === 第五组：规模（fp32 dim=0 对齐）===
    ("c13", np.float32, None,   [200, 512],   0, 500,  REPEAT,
     ["small", "atomic", "aligned", "dtype-fp32"]),
    ("c14", np.float32, None,   [5000, 2000], 0, 1000, REPEAT,
     ["large-self", "copy-bound", "atomic", "aligned", "dtype-fp32"]),
    ("c15", np.float32, None,   [200, 128],   0, 8000, REPEAT,
     ["large-M", "scatter-bound", "atomic", "aligned", "dtype-fp32"]),

    # === 第六组：特殊 / 边界 ===
    ("c16", np.float32, None,   [10000],      0, 8000, REPEAT,
     ["1d", "scalar-scatter", "dim-head", "owned-rmw", "dtype-fp32"]),  # afterDim=1
    ("c17", np.float16, None,   [4, 5, 6, 7, 100], 4, 50, REPEAT,
     ["5d", "dim-tail", "scalar-scatter", "owned-rmw", "dtype-fp16"]),  # afterDim=1
    ("c18", np.int8,    None,   [256, 512],   0, 500,  UNIQUE,
     ["dtype-int8", "atomic", "aligned", "cast", "index-unique"]),  # 512%32=0
    ("c19", np.float32, "bf16", [64, 1008],   0, 400,  REPEAT,
     ["dtype-bf16", "owned-rmw", "aligned", "cast"]),  # 1008*2=2016,%32=0
    ("c20", np.float32, None,   [100, 10000], 0, 1000, REPEAT,
     ["large-vector", "atomic", "aligned", "copy-bound", "dtype-fp32"]),  # afterDim=10000
]

CASE_IDS = [c[0] for c in CASES]


def build_case(case_id, seed=1234):
    """返回 (input_np, index_np, source_np, dim, torch_dtype_or_None, labels)."""
    row = next(c for c in CASES if c[0] == case_id)
    _, np_dt, torch_key, self_shape, dim, m, idx_spec, labels = row
    rng = np.random.RandomState(seed + (hash(case_id) & 0x7fffffff))

    dim_len = self_shape[dim]
    index_np = _index(idx_spec, dim_len, m, rng)

    source_shape = list(self_shape)
    source_shape[dim] = m

    if np.issubdtype(np_dt, np.floating):
        input_np = rng.uniform(-1, 1, self_shape).astype(np_dt)
        source_np = rng.uniform(-1, 1, source_shape).astype(np_dt)
    else:
        lo, hi = (-50, 50) if np_dt == np.int32 else (-10, 10)
        input_np = rng.randint(lo, hi, self_shape).astype(np_dt)
        source_np = rng.randint(lo, hi, source_shape).astype(np_dt)

    torch_dt = None
    if torch_key == "bf16":
        import torch
        torch_dt = torch.bfloat16
    return input_np, index_np, source_np, dim, torch_dt, labels


def case_info(case_id):
    row = next(c for c in CASES if c[0] == case_id)
    _, np_dt, torch_key, self_shape, dim, m, idx_spec, labels = row
    dt_name = torch_key if torch_key else np.dtype(np_dt).name
    return (f"{case_id} dtype={dt_name} shape={self_shape} dim={dim} "
            f"M={m} idx={idx_spec} tags={','.join(labels)}")
