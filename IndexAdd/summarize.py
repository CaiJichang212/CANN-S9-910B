"""汇总 results.jsonl，生成 custom vs builtin 对比表与维度统计（markdown）。

用法: python3 summarize.py [results.jsonl] > report_md.md
"""
import json
import sys
from pathlib import Path
from collections import defaultdict
from perf_cases import CASES, case_info

# 910B HBM 理论带宽 (GB/s), 用于带宽利用率估算
HBM_BW_GBPS = 328.0


def load(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def self_bytes(case_row):
    _, np_dt, tk, shape, dim, m, spec, labels = case_row
    dtype_size = {"float32": 4, "bf16": 2, "float16": 2, "int32": 4, "int8": 1}[
        tk if tk else np.dtype(np_dt).name]
    n = 1
    for s in shape:
        n *= s
    return n * dtype_size, dtype_size, m


def fmt(v, p=1):
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.{p}f}"
    return str(v)


def main():
    res_path = sys.argv[1] if len(sys.argv) > 1 else "perf_out/results.jsonl"
    rows = load(res_path)
    by_case = defaultdict(dict)
    for r in rows:
        if "error" in r:
            by_case[r["case_id"]][r["mode"] + "_err"] = r["error"]
        else:
            by_case[r["case_id"]][r["mode"]] = r

    case_meta = {c[0]: c for c in CASES}

    print("## 1. 全 case 性能对比（custom vs builtin）\n")
    print("| case | dtype | shape | dim | M | idx | custom(µs) | builtin(µs) | 比(自/内) | c_MTE2% | c_MTE3% | c_VEC% | c_SCALAR% | b_MTE3% | b_VEC% | 精度 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    ratios = []
    for cid in [c[0] for c in CASES]:
        cu = by_case[cid].get("custom", {})
        bu = by_case[cid].get("builtin", {})
        meta = case_meta[cid]
        _, np_dt, tk, shape, dim, m, spec, labels = meta
        dt = tk if tk else np.dtype(np_dt).name
        c_us = cu.get("task_us")
        b_us = bu.get("task_us")
        ratio = (c_us / b_us) if (c_us and b_us) else None
        if ratio:
            ratios.append((cid, ratio, labels))
        verify = "PASS" if cu.get("task_us") else ("ERR:" + str(by_case[cid].get("custom_err", "?"))[:20])
        print(f"| {cid} | {dt} | {shape} | {dim} | {m} | {spec} | "
              f"{fmt(c_us)} | {fmt(b_us)} | {fmt(ratio, 2)} | "
              f"{fmt((cu.get('aiv_mte2_ratio') or 0)*100)} | "
              f"{fmt((cu.get('aiv_mte3_ratio') or 0)*100)} | "
              f"{fmt((cu.get('aiv_vec_ratio') or 0)*100)} | "
              f"{fmt((cu.get('aiv_scalar_ratio') or 0)*100)} | "
              f"{fmt((bu.get('aiv_mte3_ratio') or 0)*100)} | "
              f"{fmt((bu.get('aiv_vec_ratio') or 0)*100)} | {verify} |")

    # 维度统计
    print("\n## 2. 按诊断维度汇总\n")
    tag_groups = {
        "scatter 路径": ["atomic", "owned-rmw"],
        "对齐性": ["aligned", "unaligned"],
        "dtype": ["dtype-fp32", "dtype-fp16", "dtype-bf16", "dtype-int32", "dtype-int8"],
        "dim 位置": ["dim-head", "dim-mid", "dim-tail"],
        "index 重复度": ["index-unique", "index-repeat", "index-extreme"],
        "规模/主导": ["copy-bound", "scatter-bound", "large-vector", "scalar-scatter", "small"],
    }
    for gname, tags in tag_groups.items():
        print(f"\n### {gname}\n")
        print("| tag | 样本 | custom 均值(µs) | builtin 均值(µs) | 均值比(自/内) |")
        print("|---|---|---|---|---|")
        for tag in tags:
            cs, bs = [], []
            for cid, ratio, labels in ratios:
                if tag in labels:
                    cs.append(by_case[cid].get("custom", {}).get("task_us"))
                    bs.append(by_case[cid].get("builtin", {}).get("task_us"))
            cs = [x for x in cs if x]; bs = [x for x in bs if x]
            if cs and bs:
                cm = sum(cs)/len(cs); bm = sum(bs)/len(bs)
                print(f"| {tag} | {len(cs)} | {cm:.1f} | {bm:.1f} | {cm/bm:.2f} |")

    # 总览
    print("\n## 3. 总览\n")
    if ratios:
        all_ratio = [r for _, r, _ in ratios]
        geomean = 1.0
        for r in all_ratio:
            geomean *= r
        geomean = geomean ** (1.0/len(all_ratio))
        slow = sum(1 for r in all_ratio if r > 1.0)
        fast = sum(1 for r in all_ratio if r <= 1.0)
        print(f"- 测试 case 数: {len(ratios)}")
        print(f"- custom/builtin 几何平均: **{geomean:.2f}x**")
        print(f"- 自定义慢于内置: {slow} / {len(ratios)}")
        print(f"- 自定义快于或等于内置: {fast} / {len(ratios)}")
        worst = max(ratios, key=lambda x: x[1])
        print(f"- 最差 case: {worst[0]} = {worst[1]:.2f}x ({worst[2]})")


if __name__ == "__main__":
    main()
