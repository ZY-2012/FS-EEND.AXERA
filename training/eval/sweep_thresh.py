#!/usr/bin/env python3
"""帧级 DER + threshold/median 网格扫描。

用 .prob.npy（逐帧 sigmoid 概率）离线重新二值化，无需重推理。
DER 定义与 NIST/md-eval 一致：最优说话人映射下的 miss + FA + confusion。
"""
import argparse
import glob
import os
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import medfilt

RES = 0.01  # 评分网格 10 ms


def load_ref(rttm_path, n_frames):
    """RTTM -> (n_frames, n_spk) bool 网格 + 边界位置"""
    spks, segs = {}, []
    for line in Path(rttm_path).read_text().splitlines():
        f = line.split()
        if 'SPEAKER' not in f:
            continue
        i = f.index('SPEAKER')
        st, du, spk = float(f[i + 3]), float(f[i + 4]), f[i + 7]
        if du <= 0:
            continue
        spks.setdefault(spk, len(spks))
        segs.append((st, st + du, spks[spk]))
    grid = np.zeros((n_frames, max(len(spks), 1)), dtype=bool)
    bounds = []
    for st, en, s in segs:
        a, b = int(round(st / RES)), min(int(round(en / RES)), n_frames)
        if b > a:
            grid[a:b, s] = True
            bounds += [a, b]
    return grid, np.array(bounds, dtype=int)


def prob_to_hyp(prob, threshold, median, smooth_probs, upsample):
    """概率 -> (n_frames, n_spk) bool 网格"""
    p = prob.astype(np.float32)
    if smooth_probs and median > 1:
        p = medfilt(p, (median, 1))
        b = p > threshold
    else:
        b = p > threshold
        if median > 1:
            b = medfilt(b.astype(np.float32), (median, 1)) > 0.5
    return np.repeat(b, upsample, axis=0)


def der(ref, hyp, bounds, collar):
    """帧级 DER（最优映射）。collar 为总宽度，即 ±collar/2。"""
    n = min(len(ref), len(hyp))
    ref, hyp = ref[:n], hyp[:n]
    keep = np.ones(n, dtype=bool)
    if collar > 0 and len(bounds):
        half = int(round(collar / 2 / RES))
        for b in bounds:
            keep[max(0, b - half):min(n, b + half)] = False
    ref, hyp = ref[keep], hyp[keep]
    n_ref, n_hyp = ref.sum(1), hyp.sum(1)
    total = n_ref.sum()
    if total == 0:
        return None
    # 最优映射：最大化同时激活的帧数
    ov = ref.astype(np.int32).T @ hyp.astype(np.int32)
    r_idx, h_idx = linear_sum_assignment(-ov)
    correct = (ref[:, r_idx] & hyp[:, h_idx]).sum(1)
    miss = np.maximum(0, n_ref - n_hyp).sum()
    fa = np.maximum(0, n_hyp - n_ref).sum()
    conf = (np.minimum(n_ref, n_hyp) - correct).sum()
    return dict(total=float(total) * RES, miss=float(miss) * RES,
                fa=float(fa) * RES, conf=float(conf) * RES)


def score(pairs, threshold, median, collar, smooth_probs, upsample):
    agg = dict(total=0.0, miss=0.0, fa=0.0, conf=0.0)
    for prob_path, ref_path in pairs:
        prob = np.load(prob_path)
        hyp = prob_to_hyp(prob, threshold, median, smooth_probs, upsample)
        ref, bounds = load_ref(ref_path, len(hyp))
        d = der(ref, hyp, bounds, collar)
        if d is None:
            continue
        for k in agg:
            agg[k] += d[k]
    if agg['total'] == 0:
        return None
    agg['der'] = (agg['miss'] + agg['fa'] + agg['conf']) / agg['total']
    for k in ('miss', 'fa', 'conf'):
        agg[k + '_r'] = agg[k] / agg['total']
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prob_dir', required=True)
    ap.add_argument('--ref_dir', required=True)
    ap.add_argument('--collar', type=float, default=0.0)
    ap.add_argument('--upsample', type=int, default=10, help='模型帧(100ms)/评分帧(10ms)')
    ap.add_argument('--only', default='', help='逗号分隔的会议名白名单')
    ap.add_argument('--thresholds', default='0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8')
    ap.add_argument('--medians', default='1,5,11,15,21,31')
    args = ap.parse_args()

    only = set(x for x in args.only.split(',') if x)
    pairs = []
    for p in sorted(glob.glob(os.path.join(args.prob_dir, '*.prob.npy'))):
        m = Path(p).name.replace('.prob.npy', '').replace('_8k', '')
        if only and m not in only:
            continue
        ref = Path(args.ref_dir) / f'{m}.rttm'
        if ref.exists():
            pairs.append((p, str(ref)))
    print(f'{len(pairs)} recordings\n')

    best = None
    for smooth in (False, True):
        for med in [int(x) for x in args.medians.split(',')]:
            row = []
            for th in [float(x) for x in args.thresholds.split(',')]:
                a = score(pairs, th, med, args.collar, smooth, args.upsample)
                if a is None:
                    continue
                row.append((th, a))
                if best is None or a['der'] < best[0]['der']:
                    best = (a, th, med, smooth)
            tag = 'prob-smooth' if smooth else 'bin-smooth '
            print(f'{tag} median={med:<3d} ' +
                  '  '.join(f'th{th:.2f}:{a["der"]:.2%}' for th, a in row))
    a, th, med, smooth = best
    print(f'\n最优: threshold={th} median={med} '
          f'{"概率域平滑" if smooth else "二值域平滑"}')
    print(f'  DER {a["der"]:.2%}  miss {a["miss_r"]:.2%}  FA {a["fa_r"]:.2%}  conf {a["conf_r"]:.2%}')


if __name__ == '__main__':
    main()
