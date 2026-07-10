#!/usr/bin/env python3
"""Aggregate Concat AICore timing from op_summary*.csv (filter aclnnMul).
Prints median Task Duration(us) per run + per-config summary."""
import csv, sys
from pathlib import Path
import numpy as np

def median_concat_durations():
    times = []
    for f in Path('./').rglob('op_summary*.csv'):
        with open(f) as fh:
            for row in csv.DictReader(fh):
                op = row.get('Op Name','')
                if 'aclnnMul' in op:
                    continue
                t = row.get('Task Duration(us)')
                try:
                    times.append(float(t))
                except (ValueError, TypeError):
                    continue
    if not times:
        return None
    return float(np.median(times))

if __name__ == '__main__':
    m = median_concat_durations()
    print(f"{int(m) if m and m==int(m) else (round(m,3) if m else 0)}")
