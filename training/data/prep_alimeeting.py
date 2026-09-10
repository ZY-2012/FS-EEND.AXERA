#!/usr/bin/env python3
"""AliMeeting far 训练集 → kaldi 风格数据目录（8 kHz 单声道 + 说话人标注）。

对每场会议：
- 8 通道 wav 取 ch0，16k -> 8k（训练管线不做重采样）
- TextGrid 的说话人 tier → segments / utt2spk / spk2utt / reco2num_spk / utt2timestamp
"""
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path('/root/autodl-tmp/alimeeting/Train_Ali_far')
OUT = Path('/root/autodl-tmp/data/alimeeting')
TG_PAT = re.compile(r'name = "([^"]+)"')
XM_PAT = re.compile(r'xmin = ([\d.]+)|xmax = ([\d.]+)')


def parse_textgrid(path):
    tiers = {}
    cur = None
    for line in path.read_text(errors='replace').splitlines():
        m = TG_PAT.search(line)
        if m and '_SPK' in m.group(1):
            cur = m.group(1)
            tiers[cur] = []
        elif cur:
            sline = line.strip()
            if sline.startswith('xmin = '):
                tiers[cur].append([float(sline[7:]), None, None])
            elif sline.startswith('xmax = '):
                tiers[cur][-1][1] = float(sline[7:])
            elif sline.startswith('text = '):
                tiers[cur][-1][2] = sline[7:].strip('"')
    return tiers


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'audio').mkdir(exist_ok=True)
    seg_lines, utt2spk, wav_lines, reco2dur, spk2utt, reco2num, utt2ts = [], [], [], [], {}, [], []
    import librosa
    wavs = sorted((ROOT / 'audio_dir').glob('*.wav'))
    print(f'{len(wavs)} meetings', flush=True)
    for wi, wav_path in enumerate(wavs):
        mid = wav_path.name.split('.')[0]
        # TextGrid 名字不带麦克风阵列后缀（R0003_M0046.TextGrid vs R0003_M0046_MS002.wav）
        tg = ROOT / 'textgrid_dir' / f'{"_".join(mid.split("_")[:2])}.TextGrid'
        if not tg.exists():
            print(f'  skip {mid}: no TextGrid', flush=True)
            continue
        # ch0 -> 8k 单声道
        data, sr = sf.read(str(wav_path), dtype='float32', always_2d=True)
        mono = data[:, 0]
        if sr != 8000:
            mono = librosa.resample(mono, orig_sr=sr, target_sr=8000)
        out_wav = OUT / 'audio' / f'{mid}.wav'
        if not out_wav.exists() or out_wav.stat().st_size == 0:
            sf.write(str(out_wav), mono, 8000, subtype='PCM_16')
        wav_lines.append(f'{mid} {out_wav}')
        reco2dur.append(f'{mid} {len(mono)/8000:.4f}')

        tiers = parse_textgrid(tg)
        spks = sorted(tiers)
        n_spk = len(spks)
        reco2num.append(f'{mid} {n_spk}')
        idx = 0
        for spk in spks:
            spk2utt.setdefault(f'{mid}_{spk}', [])
            for xmin, xmax, text in tiers[spk]:
                if not text or xmax <= xmin:
                    continue
                utt = f'{mid}_{spk}_{idx:06d}'
                idx += 1
                seg_lines.append(f'{utt} {mid} {xmin:.4f} {xmax:.4f}')
                utt2spk.append(f'{utt} {mid}_{spk}')
                utt2ts.append(f'{utt} {xmin:.4f} {xmax:.4f}')
                spk2utt[f'{mid}_{spk}'].append(utt)
        if (wi + 1) % 20 == 0:
            print(f'  {wi+1}/{len(wavs)}', flush=True)

    with open(OUT / 'segments', 'w') as f:
        f.write('\n'.join(seg_lines) + '\n')
    with open(OUT / 'utt2spk', 'w') as f:
        f.write('\n'.join(utt2spk) + '\n')
    with open(OUT / 'wav.scp', 'w') as f:
        f.write('\n'.join(wav_lines) + '\n')
    with open(OUT / 'reco2dur', 'w') as f:
        f.write('\n'.join(reco2dur) + '\n')
    with open(OUT / 'spk2utt', 'w') as f:
        for k, v in spk2utt.items():
            f.write(f'{k} {" ".join(v)}\n')
    with open(OUT / 'reco2num_spk', 'w') as f:
        f.write('\n'.join(reco2num) + '\n')
    with open(OUT / 'utt2timestamp', 'w') as f:
        f.write('\n'.join(utt2ts) + '\n')
    print(f'DONE: {len(wav_lines)} recordings, {len(seg_lines)} segments -> {OUT}')


if __name__ == '__main__':
    main()
