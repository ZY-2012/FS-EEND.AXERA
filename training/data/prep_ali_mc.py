#!/usr/bin/env python3
"""AliMeeting 多通道 → kaldi 数据目录。

每个会议的每个麦克风通道作为独立录音（rec id = {mid}_c{ch}），标注完全复用——
同一段对话经过不同房间冲激响应，是零标注成本的声学增强。

默认只用 c0/c2/c4/c6：环形阵列上相邻麦克风高度相关，取间隔通道能以一半代价
拿到大部分多样性收益，同时把 epoch 规模控制在可训练范围。
"""
import argparse
from pathlib import Path

import soundfile as sf

SRC_AUDIO = Path('/root/autodl-tmp/data/ali_mc/audio')
BASE = Path('/root/autodl-tmp/data/alimeeting')     # 单通道版，提供标注
OUT = Path('/root/autodl-tmp/data/ali_mc')


def read(p):
    return p.read_text().splitlines() if p.exists() else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--channels', default='0,2,4,6')
    args = ap.parse_args()
    chans = [int(c) for c in args.channels.split(',')]

    # 原始标注（按会议聚合）
    segs_by_rec = {}
    for l in read(BASE / 'segments'):
        f = l.split()
        segs_by_rec.setdefault(f[1], []).append((f[0], float(f[2]), float(f[3])))
    utt2spk = dict(l.split() for l in read(BASE / 'utt2spk') if l.split())
    n_spk = {l.split()[0]: l.split()[1] for l in read(BASE / 'reco2num_spk')}

    seg, u2s, scp, r2d, r2n, u2t = [], [], [], [], [], []
    s2u = {}
    n_rec = 0
    for mid, items in sorted(segs_by_rec.items()):
        for ch in chans:
            wav = SRC_AUDIO / f'{mid}_c{ch}.wav'
            if not wav.exists():
                continue
            rec = f'{mid}_c{ch}'
            dur = sf.info(str(wav)).duration
            span = max(e for _, _, e in items)
            if dur < span * 0.95:
                print(f'  DROP {rec}: 音频 {dur/60:.1f}min < 标注 {span/60:.1f}min')
                continue
            scp.append(f'{rec} {wav}')
            r2d.append(f'{rec} {dur:.4f}')
            r2n.append(f'{rec} {n_spk.get(mid, 4)}')
            for utt, st, en in items:
                if en > dur:
                    en = dur
                if en <= st:
                    continue
                u = f'{utt}_c{ch}'
                spk = f'{utt2spk.get(utt, "unk")}_c{ch}'   # 通道内说话人独立编号
                seg.append(f'{u} {rec} {st:.4f} {en:.4f}')
                u2s.append(f'{u} {spk}')
                u2t.append(f'{u} {st:.4f} {en:.4f}')
                s2u.setdefault(spk, []).append(u)
            n_rec += 1
        if n_rec % 200 == 0 and n_rec:
            print(f'  {n_rec} recs', flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'segments').write_text('\n'.join(seg) + '\n')
    (OUT / 'utt2spk').write_text('\n'.join(u2s) + '\n')
    (OUT / 'wav.scp').write_text('\n'.join(scp) + '\n')
    (OUT / 'reco2dur').write_text('\n'.join(r2d) + '\n')
    (OUT / 'reco2num_spk').write_text('\n'.join(r2n) + '\n')
    (OUT / 'utt2timestamp').write_text('\n'.join(u2t) + '\n')
    (OUT / 'spk2utt').write_text(
        '\n'.join(f'{k} {" ".join(v)}' for k, v in sorted(s2u.items())) + '\n')
    hrs = sum(float(l.split()[1]) for l in r2d) / 3600
    print(f'DONE: {n_rec} recs（{len(chans)} 通道）, {len(seg)} segs, {hrs:.1f}h -> {OUT}')


if __name__ == '__main__':
    main()
