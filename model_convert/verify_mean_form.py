#!/usr/bin/env python3
"""Check the bounded mean-form retention state against the upstream sqrt-form.

Runs the exported StreamingStep (mean form, driven by inv_count = 1/t) over the
full recording and compares its per-frame logits with the untouched upstream
recurrence, plus reports the state magnitude range that quantization must cover.
"""
import sys
from pathlib import Path
import numpy as np
import torch

from paths import LS, OUT, WAV, add_upstream_to_syspath
add_upstream_to_syspath()

DELAY = 9


def main():
    from export_streaming_step import load, StreamingStep
    from datasets.feature import extract_fbank

    model, cfg = load()
    max_nspks = cfg['data']['max_speakers'] + 2   # simu=10, AMI=6, CALLHOME=9, DIHARD=12
    step = StreamingStep(model, max_nspks).eval()
    feat = extract_fbank(str(WAV),
                         context_size=cfg['data']['context_recp'], input_transform=cfg['data']['feat_type'],
                         frame_size=cfg['data']['feat']['win_length'], frame_shift=cfg['data']['feat']['hop_length'],
                         subsampling=cfg['data']['subsampling'])
    B, H, heads, D = 1, model.n_units, step.heads, step.h // step.heads

    enc_states = []
    for _ in range(4):
        enc_states += [torch.zeros(B, heads, D, D), torch.zeros(B, H, step.convk - 1)]
    dec_states = [torch.zeros(B*max_nspks, heads, D, D) for _ in range(2)]
    conv_cache = torch.zeros(B, H, step.outk - 1)

    preds = []
    ranges = {}
    dec_t = 0
    with torch.no_grad():
        for i in range(len(feat)):
            inv = torch.full((1, heads, 1, 1), 1.0/(i+1), dtype=torch.float32)
            dec_inv = torch.full((1, heads, 1, 1), 1.0/max(dec_t+1, 1), dtype=torch.float32)
            args = [feat[i:i+1].view(1, 1, -1), inv, dec_inv, conv_cache] + enc_states + dec_states
            out = step(*args)
            for nm, v in (('enc0_kv', enc_states[0]), ('enc3_kv', enc_states[6]),
                          ('dec0_kv', dec_states[0]), ('dec1_kv', dec_states[1])):
                lo, hi = float(v.min()), float(v.max())
                if nm not in ranges:
                    ranges[nm] = [lo, hi]
                else:
                    ranges[nm][0] = min(ranges[nm][0], lo)
                    ranges[nm][1] = max(ranges[nm][1], hi)
            enc_states = []
            oi = 1
            for _ in range(4):
                enc_states += [out[oi], out[oi+1]]
                oi += 2
            conv_cache = out[oi]; oi += 1
            new_dec = [out[oi], out[oi+1]]
            if i >= DELAY:
                dec_states = new_dec
                dec_t += 1
                preds.append(out[0].numpy().astype(np.float32))

    logits = np.concatenate(preds, axis=1)[0]
    np.save(OUT/'meanform_full_pred.npy', logits)
    native = np.load(OUT/'torch_frame_streaming_pred.npy')[DELAY:DELAY+len(logits)]
    a, b = logits.reshape(-1), native.reshape(-1)
    cos = float(a @ b / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-12))
    d = np.abs(logits - native)
    print(f'frames {len(logits)}  cosine vs upstream {cos:.7f}  MAE {d.mean():.6f}  maxdiff {d.max():.6f}')
    print(f'frames with maxdiff>0.5: {(d.max(axis=1) > 0.5).sum()}/{len(logits)}')
    print('state magnitude ranges (must stay bounded):')
    for nm, (lo, hi) in ranges.items():
        print(f'  {nm}: [{lo:.3f}, {hi:.3f}]')


if __name__ == '__main__':
    main()
