#!/usr/bin/env python3
"""在指定 (threshold, median) 下打分：逐场 + 加权 + 误差分解。

用法:
  python score_at.py --prob_dir D --ref_dir R --collar 0.25 --threshold 0.55 --median 11 \
      [--only a,b,c] [--exclude x,y]
"""
import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sweep_thresh import load_ref, der, prob_to_hyp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prob_dir', required=True)
    ap.add_argument('--ref_dir', required=True)
    ap.add_argument('--collar', type=float, default=0.0)
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--median', type=int, default=11)
    ap.add_argument('--upsample', type=int, default=10)
    ap.add_argument('--only', default='')
    ap.add_argument('--exclude', default='')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    only = {x for x in args.only.split(',') if x}
    excl = {x for x in args.exclude.split(',') if x}
    agg = dict(total=0.0, miss=0.0, fa=0.0, conf=0.0)
    rows = []
    for p in sorted(glob.glob(os.path.join(args.prob_dir, '*.prob.npy'))):
        m = Path(p).name.replace('.prob.npy', '').replace('_8k', '')
        if (only and m not in only) or m in excl:
            continue
        ref_p = Path(args.ref_dir) / f'{m}.rttm'
        if not ref_p.exists():
            continue
        hyp = prob_to_hyp(np.load(p), args.threshold, args.median, False, args.upsample)
        ref, bounds = load_ref(str(ref_p), len(hyp))
        d = der(ref, hyp, bounds, args.collar)
        if d is None:
            continue
        for k in agg:
            agg[k] += d[k]
        e = d['miss'] + d['fa'] + d['conf']
        rows.append((m, e / d['total'], d['miss'] / d['total'],
                     d['fa'] / d['total'], d['conf'] / d['total'], d['total']))

    if not args.quiet:
        print(f'{"meeting":12s} {"DER":>8s} {"miss":>8s} {"FA":>8s} {"conf":>8s} {"语音":>8s}')
        for m, e, ms, fa, cf, t in rows:
            print(f'{m:12s} {e:7.2%} {ms:7.2%} {fa:7.2%} {cf:7.2%} {t/60:7.1f}m')
    e = agg['miss'] + agg['fa'] + agg['conf']
    print(f'{"加权":12s} {e/agg["total"]:7.2%} {agg["miss"]/agg["total"]:7.2%} '
          f'{agg["fa"]/agg["total"]:7.2%} {agg["conf"]/agg["total"]:7.2%} '
          f'{agg["total"]/3600:6.1f}h   (n={len(rows)}, th={args.threshold}, med={args.median})')


if __name__ == '__main__':
    main()
