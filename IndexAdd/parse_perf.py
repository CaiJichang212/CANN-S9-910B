"""解析单次 msprof 输出的 op_summary，提取 IndexAdd 的耗时与流水拆分。

用法: python3 parse_perf.py <prof_dir> <case_id> <mode> [label]
  prof_dir: msprof --output 指向的目录（其下有 PROF_GROUP_*/ */
  case_id / mode / label: 仅写入输出 JSONL 作为元信息

输出一行 JSON 到 stdout，字段:
  case_id, mode, task_us(中位数), aiv_mte2_us/ratio, aiv_mte3_us/ratio,
  aiv_vec_us/ratio, aiv_scalar_us/ratio, block_dim, input_shapes
取 [10,30) 轮 IndexAdd 记录中 Task Duration 为中位数的那条的全部流水字段。
"""
import csv
import json
import sys
from pathlib import Path
from statistics import median


FIELDS = [
    "Task Duration(us)", "Block Dim", "Input Shapes",
    "aiv_mte2_time(us)", "aiv_mte2_ratio",
    "aiv_mte3_time(us)", "aiv_mte3_ratio",
    "aiv_vec_time(us)", "aiv_vec_ratio",
    "aiv_scalar_time(us)", "aiv_scalar_ratio",
    "aiv_time(us)",
]


def _f(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def find_summary(prof_dir):
    for p in Path(prof_dir).rglob("op_summary*.csv"):
        return p
    return None


def parse(prof_dir, case_id, mode, label=""):
    summary = find_summary(prof_dir)
    if summary is None:
        return {"case_id": case_id, "mode": mode, "label": label, "error": "no op_summary"}
    rows = []
    with open(summary, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            name = r.get("Op Name", "")
            if "aclnnMul" in name or "Mul" == r.get("OP Type", ""):
                continue
            if "IndexAdd" not in name and "IndexAdd" != r.get("OP Type", ""):
                continue
            rows.append(r)
    if not rows:
        return {"case_id": case_id, "mode": mode, "label": label, "error": "no IndexAdd rows"}
    # 取稳定区间 [10, 30)
    sample = rows[10:30] if len(rows) >= 30 else rows
    durations = [_f(r["Task Duration(us)"]) for r in sample]
    durations = [d for d in durations if d is not None]
    if not durations:
        return {"case_id": case_id, "mode": mode, "label": label, "error": "no durations"}
    med = median(durations)
    # 找最接近中位数的实际记录，取其全部流水字段
    best = min(sample, key=lambda r: abs((_f(r["Task Duration(us)"]) or 0) - med))
    out = {"case_id": case_id, "mode": mode, "label": label,
           "task_us": round(med, 3), "n_samples": len(sample)}
    for fld in FIELDS[1:]:
        out[fld] = _f(best.get(fld))
    return out


if __name__ == "__main__":
    prof_dir = sys.argv[1]
    case_id = sys.argv[2]
    mode = sys.argv[3]
    label = sys.argv[4] if len(sys.argv) > 4 else ""
    print(json.dumps(parse(prof_dir, case_id, mode, label), ensure_ascii=False))
