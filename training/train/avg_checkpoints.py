#!/usr/bin/env python3
"""平均最后 N 个 epoch 的 Lightning checkpoint。

用法: python avg_checkpoints.py <ckpt_dir> <out_path> [n]
ckpt 文件须形如 epoch=<k>-step=<s>.ckpt，只取 state_dict（去掉 Lightning 元信息键）。
"""
import pathlib
import re
import sys

import torch


def main():
    ckpt_dir, out = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    d = pathlib.Path(ckpt_dir)
    fs = [p for p in d.glob('epoch=*.ckpt') if re.search(r'epoch=(\d+)', p.name)]
    fs.sort(key=lambda p: int(re.search(r'epoch=(\d+)', p.name).group(1)))
    fs = fs[-n:]
    print('averaging:', [f.name for f in fs])
    acc = None
    for f in fs:
        sd = torch.load(str(f), map_location='cpu')['state_dict']
        if acc is None:
            acc = {k: v.float() / len(fs) for k, v in sd.items()}
        else:
            for k, v in sd.items():
                acc[k] += v.float() / len(fs)
    torch.save(acc, out)
    print(f'saved {out} ({len(acc)} keys)')


if __name__ == '__main__':
    main()
