#!/usr/bin/env python3
"""AMI Mix-Headset + 官方 segments 标注 → kaldi 风格数据目录。

segments XML 解析：每个 <segment starttime endtime> 含 <speaker nite:id=...>。
"""
import re
import sys
from pathlib import Path

import soundfile as sf

AUDIO = Path('/root/autodl-tmp/ami/audio')
SEG_DIR = Path('/root/autodl-tmp/ami/segments')
OUT = Path('/root/autodl-tmp/data/ami')

SEG_PAT = re.compile(r'transcriber_start="([\d.]+)"[^>]*transcriber_end="([\d.]+)"')


def parse_segments_xml(path):
    """返回 [(start, end)] 列表（transcriber_start/end 属性）"""
    return [(float(m.group(1)), float(m.group(2)))
            for line in path.read_text(errors='replace').splitlines()
            if (m := SEG_PAT.search(line))]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'audio').mkdir(exist_ok=True)
    import librosa
    seg_lines, utt2spk, wav_lines, reco2dur, spk2utt, reco2num, utt2ts = [], [], [], [], {}, [], []
    done = 0
    for wav_path in sorted(AUDIO.glob('*.wav')):
        mid = wav_path.name.split('.')[0]
        xmls = sorted(SEG_DIR.glob(f'{mid}.*.segments.xml'))
        if not xmls:
            print(f'  skip {mid}: no segments xml', flush=True)
            continue
        data, sr = sf.read(str(wav_path), dtype='float32', always_2d=True)
        mono = data[:, 0]
        if sr != 8000:
            mono = librosa.resample(mono, orig_sr=sr, target_sr=8000)
        out_wav = OUT / 'audio' / f'{mid}.wav'
        sf.write(str(out_wav), mono, 8000, subtype='PCM_16')
        wav_lines.append(f'{mid} {out_wav}')
        reco2dur.append(f'{mid} {len(mono)/8000:.4f}')

        segs = []
        for xml in xmls:
            spk = xml.name.split('.')[1]  # EN2001a.A.segments.xml -> A
            for start, end in parse_segments_xml(xml):
                segs.append((start, end, spk))
        if not segs:
            print(f'  skip {mid}: empty segments', flush=True)
            continue
        spks = sorted(set(s for _, _, s in segs))
        reco2num.append(f'{mid} {len(spks)}')
        idx = 0
        for start, end, spk in segs:
            utt = f'{mid}_{spk}_{idx:06d}'
            idx += 1
            seg_lines.append(f'{utt} {mid} {start:.4f} {end:.4f}')
            utt2spk.append(f'{utt} {mid}_{spk}')
            utt2ts.append(f'{utt} {start:.4f} {end:.4f}')
            spk2utt.setdefault(f'{mid}_{spk}', []).append(utt)
        done += 1
        if done % 20 == 0:
            print(f'  {done}', flush=True)

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
    print(f'DONE: {done} recordings, {len(seg_lines)} segments -> {OUT}')


if __name__ == '__main__':
    main()
