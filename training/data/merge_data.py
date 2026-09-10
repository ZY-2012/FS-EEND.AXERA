#!/usr/bin/env python3
"""合并 AliMeeting 与 AMI 的 kaldi 数据目录，并切出验证集。

- /root/autodl-tmp/data/combined:   AMI 训练 + AliMeeting 训练（去掉 held-out val 会议）
- /root/autodl-tmp/data/val:        AMI dev（16 场）+ AliMeeting held-out（10 场）
"""
from pathlib import Path

ALI = Path('/root/autodl-tmp/data/alimeeting')
AMI = Path('/root/autodl-tmp/data/ami')
COMB = Path('/root/autodl-tmp/data/combined')
VAL = Path('/root/autodl-tmp/data/val')

ALI_VAL_RECS = None  # 在 main() 里按 wav.scp 顺序取前 10 场
AMI_DEV = {'ES2004a','ES2004b','ES2004c','ES2004d','IS1009a','IS1009b','IS1009c','IS1009d',
           'TS3003a','TS3003b','TS3003c','TS3003d','EN2002a','EN2002b','EN2002c','EN2002d'}


def read(path):
    return path.read_text().splitlines() if path.exists() else []


def merge(parts, out, drop_recs=set()):
    out.mkdir(parents=True, exist_ok=True)
    for name in ['segments', 'utt2spk', 'wav.scp', 'reco2dur', 'spk2utt', 'reco2num_spk', 'utt2timestamp']:
        lines = []
        for src in parts:
            for line in read(src / name):
                f = line.split()
                if not f:
                    continue
                if name in ('segments', 'utt2timestamp'):
                    rec = f[1]
                elif name == 'wav.scp':
                    rec = f[0]
                elif name == 'utt2spk':
                    # utt 形如 <rec>_<spk>_<idx>，直接由 segments 过滤即可，这里用 rec 前缀匹配
                    continue
                else:
                    rec = f[0]
                if rec in drop_recs:
                    continue
                lines.append(line)
        # utt2spk / spk2utt 由 segments + utt2spk 过滤
        (out / name).write_text('\n'.join(lines) + '\n')
    # utt2spk: 只保留出现在 segments 里的 utt
    utts = {l.split()[0] for l in read(out / 'segments')}
    lines = [l for l in (read(ALI / 'utt2spk') + read(AMI / 'utt2spk'))
             if l.split() and l.split()[0] in utts]
    (out / 'utt2spk').write_text('\n'.join(lines) + '\n')
    # spk2utt: 重建
    spk2utt = {}
    for l in lines:
        u, s = l.split()
        spk2utt.setdefault(s, []).append(u)
    (out / 'spk2utt').write_text(
        '\n'.join(f'{s} {" ".join(v)}' for s, v in sorted(spk2utt.items())) + '\n')
    print(f'{out}: {len(read(out / "wav.scp"))} recs, {len(utts)} utts')


def main():
    global ALI_VAL_RECS
    ALI_VAL_RECS = {l.split()[0] for l in read(ALI / 'wav.scp')[:10]}
    # 训练集 = AMI(train) + AliMeeting(train - val 会议)
    merge([ALI, AMI], COMB, drop_recs=ALI_VAL_RECS | AMI_DEV)
    # 验证集 = AMI dev + AliMeeting held-out
    all_recs = {l.split()[0] for l in read(ALI / 'wav.scp')} | {l.split()[0] for l in read(AMI / 'wav.scp')}
    merge([ALI, AMI], VAL, drop_recs=all_recs - (ALI_VAL_RECS | AMI_DEV))


if __name__ == '__main__':
    main()
