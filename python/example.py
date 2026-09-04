#!/usr/bin/env python3
"""LS-EEND streaming speaker diarization on AX650N: wav -> RTTM."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ls_eend_sdk import diarize  # noqa: E402


def main():
    here = Path(__file__).resolve().parent
    default_model = here.parent / 'models' / 'simu' / 'streaming_step.axmodel'

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wav', required=True, help='input audio (any sample rate)')
    parser.add_argument('--model', default=str(default_model),
                        help='models/<variant>/streaming_step.axmodel, or a .onnx for host testing')
    parser.add_argument('--rttm', default=None, help='output RTTM path (default: <wav>.rttm)')
    parser.add_argument('--max-speakers', type=int, default=8,
                        help='keep the first N of 8 speaker channels')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--median', type=int, default=11,
                        help='median filter length in frames (11 matches upstream)')
    parser.add_argument('--progress', type=int, default=0,
                        help='print progress every N frames (0 = quiet)')
    args = parser.parse_args()

    result = diarize(
        args.wav, args.model, rttm_path=args.rttm, max_speakers=args.max_speakers,
        threshold=args.threshold, median=args.median,
        progress=args.progress or None,
    )

    report = {k: v for k, v in result.items() if k not in ('logits', 'segments')}
    report['segments'] = len(result['segments'])
    print(json.dumps(report, indent=2))
    for start, end, spk in result['segments'][:10]:
        print(f'  {start:7.2f} - {end:7.2f}  speaker_{spk}')
    if len(result['segments']) > 10:
        print(f'  ... {len(result["segments"]) - 10} more')


if __name__ == '__main__':
    main()
