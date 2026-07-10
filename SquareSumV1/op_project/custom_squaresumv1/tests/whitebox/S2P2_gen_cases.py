#!/usr/bin/env python3
"""SquareSumV1 S2P2 -> S2P2_cases.json（统一 dict 解包，无通用引擎依赖）

Usage: python3 S2P2_gen_cases.py
Output: S2P2_cases.json
"""

import json, os, random

random.seed(42)
K = 2   # 每个 per_dtype pair 搭配的 group 级维度值数
OUT = os.path.join(os.path.dirname(__file__), "S2P2_cases.json")

# ── per_dtype 维度定义（每维独立 list）─────────────────────────────

# G0 (EMPTY): degenerate, rLength=0 only
_DIM_G0_fp16_rLength = [{"rLength": 0}]
_DIM_G0_fp32_rLength = [{"rLength": 0}]
_DIM_G0_bf16_rLength = [{"rLength": 0}]

# G1 (AR_FULLLOAD): rLength per dtype
_DIM_G1_fp16_rLength = [{"rLength": 1}, {"rLength": 256}, {"rLength": 4096}, {"rLength": 8192}, {"rLength": 24368}]
_DIM_G1_fp32_rLength = [{"rLength": 1}, {"rLength": 256}, {"rLength": 4096}, {"rLength": 8192}, {"rLength": 24376}]
_DIM_G1_bf16_rLength = [{"rLength": 1}, {"rLength": 256}, {"rLength": 4096}, {"rLength": 8192}, {"rLength": 24368}]

# G2 (AR_COLSPLIT): rLength per dtype
_DIM_G2_fp16_rLength = [{"rLength": 24576}, {"rLength": 65536}, {"rLength": 1000000}, {"rLength": 10000000}, {"rLength": 100000000}]
_DIM_G2_fp32_rLength = [{"rLength": 24576}, {"rLength": 65536}, {"rLength": 1000000}, {"rLength": 10000000}, {"rLength": 100000000}]
_DIM_G2_bf16_rLength = [{"rLength": 24576}, {"rLength": 65536}, {"rLength": 1000000}, {"rLength": 10000000}, {"rLength": 100000000}]

# G3 (ARA_FULLLOAD): rLength per dtype
_DIM_G3_fp16_rLength = [{"rLength": 1}, {"rLength": 128}, {"rLength": 512}, {"rLength": 1024}, {"rLength": 2047}]
_DIM_G3_fp32_rLength = [{"rLength": 1}, {"rLength": 128}, {"rLength": 1024}, {"rLength": 2048}, {"rLength": 3070}]
_DIM_G3_bf16_rLength = [{"rLength": 1}, {"rLength": 128}, {"rLength": 512}, {"rLength": 1024}, {"rLength": 2047}]

# G4 (ARA_ROWSPLIT): rLength per dtype
_DIM_G4_fp16_rLength = [{"rLength": 4095}, {"rLength": 5120}, {"rLength": 6144}, {"rLength": 8192}, {"rLength": 10000}]
_DIM_G4_fp32_rLength = [{"rLength": 6142}, {"rLength": 7168}, {"rLength": 8192}, {"rLength": 9216}, {"rLength": 10000}]
_DIM_G4_bf16_rLength = [{"rLength": 4095}, {"rLength": 5120}, {"rLength": 6144}, {"rLength": 8192}, {"rLength": 10000}]

# G5 (MULTI_AXIS): rLength per dtype
_DIM_G5_fp16_rLength = [{"rLength": 1}, {"rLength": 8}, {"rLength": 128}, {"rLength": 1024}, {"rLength": 10000}]
_DIM_G5_fp32_rLength = [{"rLength": 1}, {"rLength": 8}, {"rLength": 128}, {"rLength": 1024}, {"rLength": 10000}]
_DIM_G5_bf16_rLength = [{"rLength": 1}, {"rLength": 8}, {"rLength": 128}, {"rLength": 1024}, {"rLength": 10000}]

# ── group 级维度定义（每维独立 list）─────────────────────────────

# G0: no group-level dimensions (degenerate)

# G1: totalRows
_POOL_dim_totalRows_G1 = [{"totalRows": 1}, {"totalRows": 6}, {"totalRows": 7}, {"totalRows": 97}, {"totalRows": 100}, {"totalRows": 512}, {"totalRows": 997}, {"totalRows": 1000}, {"totalRows": 10000}, {"totalRows": 65536}]

# G2: totalRows
_POOL_dim_totalRows_G2 = [{"totalRows": 1}, {"totalRows": 11}, {"totalRows": 100}, {"totalRows": 101}, {"totalRows": 503}, {"totalRows": 777}, {"totalRows": 1024}, {"totalRows": 5000}, {"totalRows": 30000}, {"totalRows": 65536}]

# G3: totalRows + a0Length
_POOL_dim_totalRows_G3 = [{"totalRows": 1}, {"totalRows": 7}, {"totalRows": 100}, {"totalRows": 127}, {"totalRows": 256}, {"totalRows": 991}, {"totalRows": 1000}, {"totalRows": 5000}, {"totalRows": 10000}, {"totalRows": 65536}]
_POOL_dim_a0Length_G3 = [{"a0Length": 1}, {"a0Length": 7}, {"a0Length": 8}, {"a0Length": 97}, {"a0Length": 100}, {"a0Length": 128}, {"a0Length": 997}, {"a0Length": 1000}, {"a0Length": 5000}, {"a0Length": 10000}]

# G4: totalRows + a0Length
_POOL_dim_totalRows_G4 = [{"totalRows": 1}, {"totalRows": 11}, {"totalRows": 64}, {"totalRows": 109}, {"totalRows": 500}, {"totalRows": 823}, {"totalRows": 2000}, {"totalRows": 8000}, {"totalRows": 30000}, {"totalRows": 65536}]
_POOL_dim_a0Length_G4 = [{"a0Length": 1}, {"a0Length": 13}, {"a0Length": 16}, {"a0Length": 100}, {"a0Length": 211}, {"a0Length": 256}, {"a0Length": 997}, {"a0Length": 1000}, {"a0Length": 3000}, {"a0Length": 10000}]

# G5: totalRows
_POOL_dim_totalRows_G5 = [{"totalRows": 1}, {"totalRows": 7}, {"totalRows": 89}, {"totalRows": 100}, {"totalRows": 128}, {"totalRows": 600}, {"totalRows": 911}, {"totalRows": 5000}, {"totalRows": 20000}, {"totalRows": 65536}]


def compress_per_dtype(dim_dicts):
    """多 per_dtype 维度 → 单 dict 列表。"""
    if len(dim_dicts) == 1:
        return list(dim_dicts.values())[0]
    results, seen = [], set()
    dim_names = list(dim_dicts.keys())
    rng = random.Random()
    for i, primary_name in enumerate(dim_names):
        rng.seed(hash(primary_name) % 100000 + i * 31)
        for pv in dim_dicts[primary_name]:
            combo = dict(pv)
            for other_name in dim_names:
                if other_name == primary_name:
                    continue
                combo.update(rng.choice(dim_dicts[other_name]))
            key = tuple(sorted(combo.items()))
            if key not in seen:
                seen.add(key)
                results.append(combo)
    return results


def compress_group_pool(dim_dicts):
    """多 group 级维度 → 单 POOL。"""
    if len(dim_dicts) == 1:
        return list(dim_dicts.values())[0]
    rng = random.Random()
    shuffled = {}
    min_len = min(len(v) for v in dim_dicts.values())
    for name, values in dim_dicts.items():
        rng.seed(hash(name) % 100000)
        s = values[:]
        rng.shuffle(s)
        shuffled[name] = s[:min_len]
    results = []
    for i in range(min_len):
        combo = {}
        for name in shuffled:
            combo.update(shuffled[name][i])
        results.append(combo)
    return results


def shuffled_pool(base, seed):
    """返回打乱后的池和位置指针。"""
    rng = random.Random(seed)
    p = base[:]
    rng.shuffle(p)
    return p, 0


# ── per_dtype 压缩 ──────────────────────────────────────────────

# G0: single dim, direct assign
G0_fp16 = _DIM_G0_fp16_rLength
G0_fp32 = _DIM_G0_fp32_rLength
G0_bf16 = _DIM_G0_bf16_rLength

# G1: single dim, direct assign
G1_fp16 = _DIM_G1_fp16_rLength
G1_fp32 = _DIM_G1_fp32_rLength
G1_bf16 = _DIM_G1_bf16_rLength

# G2: single dim, direct assign
G2_fp16 = _DIM_G2_fp16_rLength
G2_fp32 = _DIM_G2_fp32_rLength
G2_bf16 = _DIM_G2_bf16_rLength

# G3: single dim, direct assign
G3_fp16 = _DIM_G3_fp16_rLength
G3_fp32 = _DIM_G3_fp32_rLength
G3_bf16 = _DIM_G3_bf16_rLength

# G4: single dim, direct assign
G4_fp16 = _DIM_G4_fp16_rLength
G4_fp32 = _DIM_G4_fp32_rLength
G4_bf16 = _DIM_G4_bf16_rLength

# G5: single dim, direct assign
G5_fp16 = _DIM_G5_fp16_rLength
G5_fp32 = _DIM_G5_fp32_rLength
G5_bf16 = _DIM_G5_bf16_rLength

# ── group 级 POOL 压缩 ──────────────────────────────────────────

POOL_G1 = compress_group_pool({"totalRows": _POOL_dim_totalRows_G1})
POOL_G2 = compress_group_pool({"totalRows": _POOL_dim_totalRows_G2})
POOL_G3 = compress_group_pool({"totalRows": _POOL_dim_totalRows_G3, "a0Length": _POOL_dim_a0Length_G3})
POOL_G4 = compress_group_pool({"totalRows": _POOL_dim_totalRows_G4, "a0Length": _POOL_dim_a0Length_G4})
POOL_G5 = compress_group_pool({"totalRows": _POOL_dim_totalRows_G5})

# ── group: G0 (Mode C — no group-level dims, degenerate) ────────

g_G0 = []
for dtype_entries in [("float16", G0_fp16), ("float", G0_fp32), ("bfloat16", G0_bf16)]:
    dtype_name, pairs = dtype_entries
    for p in pairs:
        g_G0.append({"_group": "G0", "dtype": dtype_name, **p})

# ── group: G1 (Mode A — multi-dtype with pool) ─────────────────

g_G1 = []
pool, pos = shuffled_pool(POOL_G1, 101)
for dtype_entries in [("float16", G1_fp16), ("float", G1_fp32), ("bfloat16", G1_bf16)]:
    dtype_name, pairs = dtype_entries
    for p in pairs:
        for _ in range(K):
            if pos >= len(pool):
                pool, pos = shuffled_pool(POOL_G1, 101)
            gp = pool[pos]; pos += 1
            g_G1.append({"_group": "G1", "dtype": dtype_name, **p, **gp})

# ── group: G2 (Mode A — multi-dtype with pool) ─────────────────

g_G2 = []
pool, pos = shuffled_pool(POOL_G2, 202)
for dtype_entries in [("float16", G2_fp16), ("float", G2_fp32), ("bfloat16", G2_bf16)]:
    dtype_name, pairs = dtype_entries
    for p in pairs:
        for _ in range(K):
            if pos >= len(pool):
                pool, pos = shuffled_pool(POOL_G2, 202)
            gp = pool[pos]; pos += 1
            g_G2.append({"_group": "G2", "dtype": dtype_name, **p, **gp})

# ── group: G3 (Mode A — multi-dtype with pool) ─────────────────

g_G3 = []
pool, pos = shuffled_pool(POOL_G3, 303)
for dtype_entries in [("float16", G3_fp16), ("float", G3_fp32), ("bfloat16", G3_bf16)]:
    dtype_name, pairs = dtype_entries
    for p in pairs:
        for _ in range(K):
            if pos >= len(pool):
                pool, pos = shuffled_pool(POOL_G3, 303)
            gp = pool[pos]; pos += 1
            g_G3.append({"_group": "G3", "dtype": dtype_name, **p, **gp})

# ── group: G4 (Mode A — multi-dtype with pool) ─────────────────

g_G4 = []
pool, pos = shuffled_pool(POOL_G4, 404)
for dtype_entries in [("float16", G4_fp16), ("float", G4_fp32), ("bfloat16", G4_bf16)]:
    dtype_name, pairs = dtype_entries
    for p in pairs:
        for _ in range(K):
            if pos >= len(pool):
                pool, pos = shuffled_pool(POOL_G4, 404)
            gp = pool[pos]; pos += 1
            g_G4.append({"_group": "G4", "dtype": dtype_name, **p, **gp})

# ── group: G5 (Mode A — multi-dtype with pool) ─────────────────

g_G5 = []
pool, pos = shuffled_pool(POOL_G5, 505)
for dtype_entries in [("float16", G5_fp16), ("float", G5_fp32), ("bfloat16", G5_bf16)]:
    dtype_name, pairs = dtype_entries
    for p in pairs:
        for _ in range(K):
            if pos >= len(pool):
                pool, pos = shuffled_pool(POOL_G5, 505)
            gp = pool[pos]; pos += 1
            g_G5.append({"_group": "G5", "dtype": dtype_name, **p, **gp})

# ── merge & dedup ────────────────────────────────────────────────

all_cases = g_G0 + g_G1 + g_G2 + g_G3 + g_G4 + g_G5

ROUTING_KEYS = ["rLength"]

seen = set()
unique = []
for c in all_cases:
    k = tuple([c["_group"], c["dtype"]] + [c[key] for key in ROUTING_KEYS])
    if k not in seen:
        seen.add(k)
        unique.append(c)

# 非路由维度默认值（所有 case 统一）
for c in unique:
    c["keep_dims"] = False

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(unique, f, indent=2, ensure_ascii=False)

# ── summary ──────────────────────────────────────────────────────
from collections import Counter
cnt = Counter(c["_group"] for c in unique)
print(f"Generated {len(unique)} cases -> {OUT}")
for g in sorted(cnt.keys()):
    print(f"  {g}: {cnt[g]}")
