#!/usr/bin/env python3
"""Full-recording ONNX streaming loop vs native PyTorch streaming logits.

Verifies the caller-side warmup gate: during the first `DELAY` frames
StreamingConv1d emits nothing, so the native loop leaves the decoder untouched.
The loop here mirrors that by discarding `pred` and restoring the previous
decoder states, so emitted frame k lines up with native output index k.
"""
import sys
from pathlib import Path
import numpy as np
import onnxruntime as ort

from paths import CONFIG_YAML, ONNX_PATH, OUT as OUT_DIR, WAV, add_upstream_to_syspath
add_upstream_to_syspath()

ONNX = ONNX_PATH
NATIVE = OUT_DIR/'torch_frame_streaming_pred.npy'
OUT = OUT_DIR/'onnx_full_pred.npy'
DELAY = 9

INPUTS = ['feat','inv_count','dec_inv_count','conv_cache'] + \
         [f'enc{i}_{x}' for i in range(4) for x in ('kv','conv')] + [f'dec{i}_kv' for i in range(2)]
OUTPUTS = ['pred'] + [f'enc{i}_{x}' for i in range(4) for x in ('inc','conv_out')] + \
          ['conv_cache_out'] + [f'dec{i}_inc' for i in range(2)]
ENC_MEANS = [f'enc{i}_kv' for i in range(4)]
DEC_MEANS = [f'dec{i}_kv' for i in range(2)]


def main():
    from datasets.feature import extract_fbank
    import hyperpyyaml
    cfgp = CONFIG_YAML
    with open(cfgp) as f:
        cfg = hyperpyyaml.load_hyperpyyaml(f)
    feat = extract_fbank(str(WAV),
                         context_size=cfg['data']['context_recp'], input_transform=cfg['data']['feat_type'],
                         frame_size=cfg['data']['feat']['win_length'], frame_shift=cfg['data']['feat']['hop_length'],
                         subsampling=cfg['data']['subsampling']).numpy().astype(np.float32)

    sess = ort.InferenceSession(str(ONNX), providers=['CPUExecutionProvider'])
    shapes = {i.name: tuple(i.shape) for i in sess.get_inputs()}
    state = {n: np.zeros(shapes[n], dtype=np.float32) for n in INPUTS}

    preds = []
    dec_t = 0
    for t in range(len(feat)):
        state['feat'] = feat[t][None, None, :]
        enc_b = 1.0/(t+1)
        dec_b = 1.0/max(dec_t+1, 1)
        state['inv_count'] = np.full(shapes['inv_count'], enc_b, dtype=np.float32)
        state['dec_inv_count'] = np.full(shapes['dec_inv_count'], dec_b, dtype=np.float32)
        out = dict(zip(OUTPUTS, sess.run(OUTPUTS, {n: np.ascontiguousarray(state[n], dtype=np.float32) for n in INPUTS})))
        # FP32 host accumulation of the retention means, mirroring the graph's
        # internal update but without a quantize/dequantize round trip per frame.
        for i, nm in enumerate(ENC_MEANS):
            state[nm] = state[nm] + (out[f'enc{i}_inc'] - state[nm]) * enc_b
            state[f'enc{i}_conv'] = out[f'enc{i}_conv_out']
        state['conv_cache'] = out['conv_cache_out']
        if t >= DELAY:
            for i, nm in enumerate(DEC_MEANS):
                state[nm] = state[nm] + (out[f'dec{i}_inc'] - state[nm]) * dec_b
            preds.append(out['pred'].astype(np.float32)); dec_t += 1
        if (t+1) % 400 == 0:
            print(f'  {t+1}/{len(feat)} frames', flush=True)

    logits = np.concatenate(preds, axis=1)[0]
    np.save(OUT, logits)
    # save_torch_streaming_logits appends a zero placeholder for every frame where
    # StreamingConv1d returned None, so real native outputs start at index DELAY.
    native = np.load(NATIVE)[DELAY:DELAY+len(logits)]
    a, b = logits.reshape(-1), native.reshape(-1)
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    d = np.abs(logits - native)
    print(f'frames {len(logits)}  cosine {cos:.7f}  MAE {d.mean():.6f}  maxdiff {d.max():.6f}')
    print(f'frames with maxdiff>0.5: {(d.max(axis=1) > 0.5).sum()}/{len(logits)}')


if __name__ == '__main__':
    main()
