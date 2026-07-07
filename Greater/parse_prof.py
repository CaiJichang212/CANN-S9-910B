"""Parse all prof_out/*/op_summary CSVs for Greater: median time + pipeline ratios."""
import csv, glob, sys, numpy as np

COLS = ['Task Duration(us)', 'aiv_vec_ratio', 'aiv_scalar_ratio',
        'aiv_mte2_ratio', 'aiv_mte3_ratio', 'aiv_vec_time(us)',
        'aiv_mte2_time(us)', 'aiv_mte3_time(us)', 'aiv_scalar_time(us)']

def parse(spec_dir):
    rows = []
    for f in glob.glob(f"{spec_dir}/**/op_summary*.csv", recursive=True):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                op = r.get('Op Name','')
                if 'Greater' not in op or 'aclnnMul' in op:
                    continue
                try:
                    dur = float(r['Task Duration(us)'])
                except:
                    continue
                rows.append(r)
    if not rows:
        return None
    # use samples 10..30 (skip warmup), like get_time.py
    samp = rows[10:30] if len(rows) >= 30 else rows
    med = lambda k: float(np.median([float(s[k]) for s in samp if s.get(k,'').strip() not in ('','N/A')])) if any(s.get(k,'').strip() not in ('','N/A') for s in samp) else 0
    return dict(dur=med('Task Duration(us)'),
                vec=med('aiv_vec_ratio'), sca=med('aiv_scalar_ratio'),
                mte2=med('aiv_mte2_ratio'), mte3=med('aiv_mte3_ratio'),
                vec_t=med('aiv_vec_time(us)'), mte2_t=med('aiv_mte2_time(us)'),
                mte3_t=med('aiv_mte3_time(us)'), sca_t=med('aiv_scalar_time(us)'),
                n=len(rows))

if __name__ == '__main__':
    import os
    base = 'prof_out'
    specs = sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base,d)))
    print(f"{'spec':<22}{'dur_us':>9}{'MTE2%':>8}{'VEC%':>7}{'SCA%':>7}{'MTE3%':>7}   bound")
    print('-'*70)
    for s in specs:
        r = parse(os.path.join(base,s))
        if not r:
            print(f"{s:<22}  (no data)"); continue
        # bound = max ratio unit
        ratios = {'MTE2':r['mte2'],'VEC':r['vec'],'SCA':r['sca'],'MTE3':r['mte3']}
        b = max(ratios, key=ratios.get)
        print(f"{s:<22}{r['dur']:9.2f}{r['mte2']*100:8.1f}{r['vec']*100:7.1f}{r['sca']*100:7.1f}{r['mte3']*100:7.1f}   {b}({ratios[b]*100:.0f}%) n={r['n']}")
