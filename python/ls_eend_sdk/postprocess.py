"""Logits -> speaker segments -> RTTM.

Matches upstream ``FS-EEND/LS-EEND/train/utils/make_rttm.py``: sigmoid, threshold
0.5, 11-frame median filter, then run-length encode each speaker channel.
"""
from __future__ import annotations

import numpy as np

from .feature import FRAME_SEC

SILENCE_CHANNEL = 0
FIRST_SPEAKER_CHANNEL = 1
LAST_SPEAKER_CHANNEL = 8  # ch9 is the non-speaker slot


def to_activity(logits, max_speakers=None, threshold=0.5, median=11):
    """logits (T,10) -> binary activity grid (T, n_speakers).

    Channel 0 is silence and channel 9 is the non-speaker slot, so only
    channels 1..8 are speakers. ``max_speakers`` keeps the first N of them.
    """
    from scipy.signal import medfilt

    probs = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float32)))
    last = LAST_SPEAKER_CHANNEL
    if max_speakers is not None:
        last = min(last, FIRST_SPEAKER_CHANNEL + max_speakers - 1)
    active = (probs[:, FIRST_SPEAKER_CHANNEL:last + 1] > threshold).astype(int)
    if median > 1:
        active = medfilt(active, (median, 1))
    return active


def to_segments(activity, frame_sec=FRAME_SEC, duration=None):
    """Binary grid -> [(start_sec, end_sec, speaker_index)], sorted by time."""
    segments = []
    for spk in range(activity.shape[1]):
        column = np.pad(activity[:, spk], (1, 1))
        changes = np.where(np.diff(column) != 0)[0]
        for start, end in zip(changes[::2], changes[1::2]):
            st, ed = start * frame_sec, end * frame_sec
            if duration is not None:
                if st >= duration:
                    continue
                ed = min(ed, duration)
            if ed > st:
                segments.append((float(st), float(ed), int(spk)))
    segments.sort(key=lambda s: (s[0], s[2]))
    return segments


def write_rttm(segments, path, uri='audio'):
    """Write NIST RTTM. Speaker labels are ``<uri>_<index>``."""
    with open(path, 'w') as handle:
        for start, end, spk in segments:
            handle.write(
                f'SPEAKER {uri} 1 {start:.3f} {end - start:.3f} '
                f'<NA> <NA> {uri}_{spk} <NA>\n'
            )
    return path
