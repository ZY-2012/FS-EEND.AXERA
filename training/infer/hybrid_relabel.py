#!/usr/bin/env python3
"""混合方案：LS-EEND 帧级边界/重叠 + CAM++ 声纹重标身份。

动机（实测）：LS-EEND round2 在 AliMeeting 上 miss 9.07%、FA 4.60% 均优于/接近
3D-Speaker +overlap（10.04%/2.91%），唯一短板是 conf 10.15% vs 5.53% —— 远场混响下
attractor 把 4 个说话人塌缩（R8007_M8010 槽位激活 [.71 .60 .17 .015]）。
本脚本保留 LS-EEND 的时间轴判决，仅把"这段是谁"交给 CAM++ 判定。

做法：
1. LS-EEND 概率 -> 二值网格（threshold/median 已调好）
2. 切成 1.5s/0.75s 的子段；只用"单说话人"子段（该时刻恰好一个槽位激活）提 CAM++ 嵌入
3. 谱聚类得到全局说话人 ID
4. 每个槽位的每一段按其覆盖的子段投票决定 ID —— 因此同一槽位内被合并的两个人可被拆开
5. 重叠区保留多个槽位各自的 ID
"""
import argparse
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torchaudio

sys.path.insert(0, '/data/shared/huyuan/vad_seg_sr_cluster/3D-Speaker')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.signal import medfilt

FRAME = 0.1          # LS-EEND 输出帧长（100 ms）
SUB_DUR, SUB_STEP = 1.5, 0.75


def slot_segments(binary):
    """(T, n_slot) bool -> {slot: [(st_frame, ed_frame)]}"""
    out = {}
    for s in range(binary.shape[1]):
        col = np.pad(binary[:, s].astype(np.int8), (1, 1))
        ch = np.flatnonzero(np.diff(col) != 0)
        segs = list(zip(ch[::2], ch[1::2]))
        if segs:
            out[s] = segs
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prob', required=True)
    ap.add_argument('--wav', required=True, help='16 kHz 单声道音频')
    ap.add_argument('--out_rttm', required=True)
    ap.add_argument('--threshold', type=float, default=0.55)
    ap.add_argument('--median', type=int, default=11)
    ap.add_argument('--speaker_num', type=int, default=None)
    ap.add_argument('--sub_dur', type=float, default=1.5)
    ap.add_argument('--sub_step', type=float, default=0.75)
    ap.add_argument('--purity', type=float, default=0.8)
    ap.add_argument('--self_emb', default=None,
                    help='给定 .emb.npy 则用 LS-EEND 自身帧嵌入替代 CAM++（纯 LS 方案）')
    ap.add_argument('--mer_cos', type=float, default=0.8)
    ap.add_argument('--max_spk', type=int, default=15)
    args = ap.parse_args()

    from speakerlab.bin.infer_diarization import (get_speaker_embedding_model,
                                                 get_cluster_backend)
    from speakerlab.utils.utils import circle_pad

    prob = np.load(args.prob).astype(np.float32)
    b = prob > args.threshold
    if args.median > 1:
        b = medfilt(b.astype(np.float32), (args.median, 1)) > 0.5
    n_active = b.sum(1)

    wav, fs = torchaudio.load(args.wav)
    if fs != 16000:
        wav = torchaudio.functional.resample(wav, fs, 16000)
        fs = 16000
    if wav.shape[0] > 1:
        wav = wav[:1]

    # 单说话人子段（用于提嵌入）
    chunks, owners = [], []
    for slot, segs in slot_segments(b).items():
        for st_f, ed_f in segs:
            st, ed = st_f * FRAME, ed_f * FRAME
            t = st
            while t + args.sub_dur < ed + args.sub_step:
                a, z = t, min(t + args.sub_dur, ed)
                if z - a < 0.5:
                    break
                fa, fz = int(a / FRAME), max(int(a / FRAME) + 1, int(z / FRAME))
                if (n_active[fa:fz] == 1).mean() > args.purity:   # 该子段基本无重叠
                    chunks.append([a, z])
                    owners.append((slot, st_f, ed_f))
                t += args.sub_step

    if len(chunks) < 4:
        print(f'[WARN] 可用单说话人子段仅 {len(chunks)} 个，退化为原始槽位标签')
        segs_map = {s: v for s, v in slot_segments(b).items()}
        lines = [f'SPEAKER {Path(args.wav).stem} 1 {st*FRAME:.3f} {(ed-st)*FRAME:.3f} '
                 f'<NA> <NA> spk{s} <NA>' for s, v in segs_map.items() for st, ed in v]
        Path(args.out_rttm).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_rttm).write_text('\n'.join(lines) + '\n')
        return

    from speakerlab.process.cluster import CommonClustering
    cluster = CommonClustering(cluster_type='spectral', mer_cos=args.mer_cos,
                               min_num_spks=1, max_num_spks=args.max_spk,
                               min_cluster_size=4, oracle_num=None, pval=0.012)

    if args.self_emb:
        # 纯 LS-EEND：用模型自身帧嵌入在子段上取均值
        fe = np.load(args.self_emb).astype(np.float32)
        fe = fe / (np.linalg.norm(fe, axis=1, keepdims=True) + 1e-8)
        embs = np.stack([fe[int(a / FRAME):max(int(a / FRAME) + 1, int(z / FRAME))].mean(0)
                         for a, z in chunks])
        embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        emb_model, feat_ext = get_speaker_embedding_model(device)
        wavs = [wav[0, int(a * fs):int(z * fs)] for a, z in chunks]
        mx = max(x.shape[0] for x in wavs)
        wavs = torch.stack([circle_pad(x, mx) for x in wavs]).unsqueeze(1)
        embs = []
        with torch.no_grad():
            for i in range(0, len(chunks), 64):
                batch = wavs[i:i + 64].to(device)
                embs.append(emb_model(torch.vmap(feat_ext)(batch)).cpu())
        embs = torch.cat(embs).numpy()

    labels = cluster(embs, speaker_num=args.speaker_num)
    n_spk = int(labels.max()) + 1

    # 每个(槽位,段)按其子段投票决定说话人 ID
    votes = defaultdict(Counter)
    for (slot, st_f, ed_f), lab in zip(owners, labels):
        votes[(slot, st_f, ed_f)][int(lab)] += 1

    # 槽位级兜底：无干净子段的段落沿用该槽位的主簇（避免凭空造出新说话人）
    slot_major = defaultdict(Counter)
    for (slot, _, _), c in votes.items():
        slot_major[slot].update(c)

    rec = Path(args.wav).stem
    lines = []
    for slot, segs in slot_segments(b).items():
        for st_f, ed_f in segs:
            v = votes.get((slot, st_f, ed_f))
            if v:
                spk = v.most_common(1)[0][0]
            elif slot_major[slot]:
                spk = slot_major[slot].most_common(1)[0][0]
            else:
                spk = slot
            lines.append(f'SPEAKER {rec} 1 {st_f*FRAME:.3f} {(ed_f-st_f)*FRAME:.3f} '
                         f'<NA> <NA> spk{spk} <NA>')
    Path(args.out_rttm).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_rttm).write_text('\n'.join(sorted(lines, key=lambda l: float(l.split()[3]))) + '\n')
    print(f'[INFO] {rec}: {len(chunks)} 子段, 聚类出 {n_spk} 人, {len(lines)} 段 -> {args.out_rttm}')


if __name__ == '__main__':
    main()
