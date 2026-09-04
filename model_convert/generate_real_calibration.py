"""Generate real streaming-state calibration samples from the test WAV.

Each sample is one actual frame and the recurrent states immediately before
that frame, collected by the same patched PyTorch streaming step used for ONNX
export. This avoids calibrating exposed state inputs with zeros/ones only.

Samples are spread evenly over the WHOLE recording, not just its head. The
retention states are cumulative sums: `dec*_scale` counts frames (1 -> 1912) and
`dec0_kv` grows to +-45k by the end. Calibrating only on early frames makes U16
MinMax clip every later frame, which is what destroyed streaming accuracy before.
Sampling also starts after the conv warmup window so decoder states are real.
"""
from pathlib import Path
import sys
import numpy as np
import torch

from paths import CALIB_DIR, WAV, add_upstream_to_syspath
add_upstream_to_syspath()
from export_streaming_step import load, StreamingStep
from datasets.feature import extract_fbank

N_SAMPLES = 64
DELAY = 9

INPUT_NAMES = ['feat', 'inv_count', 'dec_inv_count', 'conv_cache'] + \
              [f'enc{i}_{x}' for i in range(4) for x in ('kv', 'conv')] + \
              [f'dec{i}_kv' for i in range(2)]


def main():
    model, cfg = load()
    step = StreamingStep(model, 10).eval()
    feat = extract_fbank(str(WAV), context_size=cfg['data']['context_recp'], input_transform=cfg['data']['feat_type'],
                         frame_size=cfg['data']['feat']['win_length'], frame_shift=cfg['data']['feat']['hop_length'],
                         subsampling=cfg['data']['subsampling'])
    B, H, heads, D = 1, model.n_units, step.heads, step.h // step.heads
    enc_states = []
    for _ in range(4):
        enc_states += [torch.zeros(B, heads, D, D), torch.zeros(B, H, step.convk - 1)]
    dec_states = [torch.zeros(B * 10, heads, D, D) for _ in range(2)]
    conv_cache = torch.zeros(B, H, step.outk - 1)

    sample_dirs = {name: CALIB_DIR / name for name in INPUT_NAMES}
    for d in sample_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob('*.npy'):
            old.unlink()

    total = len(feat)
    # Evenly spaced frames across the recording, plus the final frame so the
    # largest state magnitudes are inside the calibrated range.
    picks = set(np.linspace(DELAY, total - 1, N_SAMPLES).round().astype(int).tolist())
    kept = 0
    dec_t = 0
    with torch.no_grad():
        for i in range(total):
            x = feat[i:i + 1].view(1, 1, -1)
            inv = torch.full((1, heads, 1, 1), 1.0 / (i + 1), dtype=torch.float32)
            dec_inv = torch.full((1, heads, 1, 1), 1.0 / max(dec_t + 1, 1), dtype=torch.float32)
            args = [x, inv, dec_inv, conv_cache] + enc_states + dec_states
            if i in picks:
                for name, value in zip(INPUT_NAMES, args):
                    np.save(sample_dirs[name] / f'{kept:04d}.npy', value.numpy().astype('float32'))
                kept += 1
            outputs = step(*args)
            enc_states = []
            oi = 1
            for _ in range(4):
                enc_states += [outputs[oi], outputs[oi + 1]]
                oi += 2
            conv_cache = outputs[oi]
            oi += 1
            new_dec = [outputs[oi], outputs[oi + 1]]
            # Native semantics: the decoder is not invoked during the conv warmup
            # (0-based frames 0..DELAY-1), so its states stay at the initial value
            # until the first real emission at frame DELAY.
            if i >= DELAY:
                dec_states = new_dec
                dec_t += 1
    print(f'generated {kept} real state samples spanning frames {DELAY}..{total-1} of {WAV}')
    print('feature shape:', tuple(feat.shape))


if __name__ == '__main__':
    main()
