"""解析 prof_matrix_out/<spec>/ 下的 op_summary CSV, 输出综合性能表。

每行: spec, dtype, shape, nelem, AICore 时间中位数 (sample[10:30]),
MTE2/VEC/Scalar/MTE3 流水占比, 主 bound 单元, 有效带宽 (GB/s), 精度。

有效带宽 = (x 字节 + y 字节 + z 字节) / 时间
  每个 tensor 的 HBM 实际移动量与广播无关 (广播操作数元素少, 自然读得少;
  UB 内驻留/批量重用不产生额外 HBM 流量), 故直接按各输入 shape 的元素数计。
"""
import csv
import glob
import os
import sys
import numpy as np

JOB_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, JOB_DIR)
import prof_matrix as M  # noqa: E402  (容器内有 custom_ops_lib)


def dt_bytes(dt):
    s = str(dt)
    if 'bfloat16' in s:          # torch.bfloat16 (dtype 实例, np.dtype 不接受)
        return 2
    return np.dtype(dt).itemsize


def dt_short(dt):
    s = str(dt)
    if 'bfloat16' in s:
        return 'bf16'
    name = np.dtype(dt).name
    return {'float16': 'f16', 'float32': 'f32', 'int32': 'i32', 'int8': 'i8'}.get(name, name)


def out_nelem(xs, ys):
    nd = max(len(xs), len(ys))
    px = [1] * (nd - len(xs)) + list(xs)
    py = [1] * (nd - len(ys)) + list(ys)
    n = 1
    for a, b in zip(px, py):
        n *= max(int(a), int(b))
    return n


def prod(shape):
    n = 1
    for d in shape:
        n *= int(d)
    return n


def parse_spec(spec, specdir):
    rows = []
    for f in glob.glob(f"{specdir}/**/op_summary*.csv", recursive=True):
        try:
            with open(f) as fh:
                for r in csv.DictReader(fh):
                    op = r.get('Op Name', '')
                    # op_summary 中 Greater 的 Op Name 就是 "Greater" (无 aclnn 前缀);
                    # 排除预热用的 aclnnMul_..._Mul。
                    if 'Greater' not in op or 'Mul' in op:
                        continue
                    try:
                        float(r['Task Duration(us)'])
                    except (KeyError, ValueError):
                        continue
                    rows.append(r)
        except Exception:
            continue
    if not rows:
        return None
    # 去掉前 20% warmup (prof_matrix 每轮内层跑 30 次, 1050 样本时去前 210),
    # 取稳态中位数; 样本不足时退化为全部。
    n = len(rows)
    samp = rows[n // 5:] if n >= 10 else rows

    def med(k):
        vals = [float(s[k]) for s in samp if s.get(k, '').strip() not in ('', 'N/A')]
        return float(np.median(vals)) if vals else 0.0

    dur = med('Task Duration(us)')
    mte2 = med('aiv_mte2_ratio')
    vec = med('aiv_vec_ratio')
    sca = med('aiv_scalar_ratio')
    mte3 = med('aiv_mte3_ratio')
    ratios = {'MTE2': mte2, 'VEC': vec, 'SCA': sca, 'MTE3': mte3}
    bound = max(ratios, key=ratios.get) if any(ratios.values()) else '-'
    return dict(dur=dur, mte2=mte2, vec=vec, sca=sca, mte3=mte3, bound=bound, n=len(rows))


def acc_from_log(specdir):
    for f in glob.glob(f"{specdir}/app.log"):
        try:
            with open(f) as fh:
                for line in fh:
                    if 'acc=' in line:
                        return 'PASS' if 'PASS' in line else 'FAIL'
        except Exception:
            pass
    return '?'


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else os.path.join(JOB_DIR, 'prof_matrix_out')
    specs = sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))

    out_csv = os.path.join(base, 'summary.csv')
    rows_out = []
    print(f"{'spec':<20}{'dtype':>5}{'nelem':>12}{'dur_us':>9}{'MTE2%':>7}{'VEC%':>7}"
          f"{'SCA%':>7}{'MTE3%':>7}  {'bound':<6}{'effBW_GBs':>11}{'acc':>5}  note")
    print('-' * 120)
    for spec in specs:
        r = parse_spec(spec, os.path.join(base, spec))
        if r is None:
            print(f"{spec:<20}  (no op_summary data)")
            continue
        xs, ys, dt, note = M.MATRIX[spec]
        db = dt_bytes(dt)
        xB = prod(xs) * db
        yB = prod(ys) * db
        zB = out_nelem(xs, ys) * 1
        bw = (xB + yB + zB) / (r['dur'] * 1e-6) / 1e9 if r['dur'] > 0 else 0.0  # GB/s
        acc = acc_from_log(os.path.join(base, spec))
        shape_str = f"{list(xs)}x{list(ys)}"
        print(f"{spec:<20}{dt_short(dt):>5}{out_nelem(xs,ys):>12}{r['dur']:9.2f}"
              f"{r['mte2']*100:7.1f}{r['vec']*100:7.1f}{r['sca']*100:7.1f}{r['mte3']*100:7.1f}"
              f"  {r['bound']:<6}{bw:11.1f}{acc:>5}  {note}")
        rows_out.append(dict(spec=spec, dtype=dt_short(dt), shape=shape_str,
                             nelem=out_nelem(xs, ys), dur_us=round(r['dur'], 2),
                             mte2_pct=round(r['mte2']*100, 1), vec_pct=round(r['vec']*100, 1),
                             sca_pct=round(r['sca']*100, 1), mte3_pct=round(r['mte3']*100, 1),
                             bound=r['bound'], eff_bw_GBs=round(bw, 1), acc=acc,
                             x_MB=round(xB/1e6, 1), y_MB=round(yB/1e6, 1), z_MB=round(zB/1e6, 1),
                             note=note, n=r['n']))

    with open(out_csv, 'w', newline='') as fh:
        if rows_out:
            w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
            w.writeheader()
            w.writerows(rows_out)
    print(f"\nwrote {out_csv} ({len(rows_out)} rows)")


if __name__ == '__main__':
    main()
