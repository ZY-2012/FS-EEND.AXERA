"""End-to-end WAV -> RTTM speaker diarization."""
from __future__ import annotations

import time
from pathlib import Path

from .feature import wav_to_features
from .postprocess import to_activity, to_segments, write_rttm
from .session import StreamingDiarizer


def diarize(wav_path, model_path, rttm_path=None, max_speakers=None,
            threshold=0.5, median=11, progress=None):
    """Run diarization on one recording.

    Args:
        wav_path: input audio (any sample rate; resampled to 8 kHz).
        model_path: ``streaming_step.axmodel`` (board) or ``.onnx`` (host).
        rttm_path: optional output path; defaults to ``<wav>.rttm``.
        max_speakers: keep only the first N speaker channels (of 8).
        threshold: activity threshold on the sigmoid.
        median: median filter length in frames (11 matches upstream).
        progress: print progress every N frames when set.

    Returns a dict with ``segments``, ``rttm``, ``logits`` and timing.
    """
    wav_path = Path(wav_path)
    features, duration = wav_to_features(wav_path)

    runner = StreamingDiarizer(model_path)
    start = time.time()
    logits = runner.run(features, progress=progress)
    elapsed_ms = (time.time() - start) * 1000.0

    activity = to_activity(logits, max_speakers=max_speakers,
                           threshold=threshold, median=median)
    segments = to_segments(activity, duration=duration)

    if rttm_path is None:
        rttm_path = wav_path.with_suffix('.rttm')
    write_rttm(segments, rttm_path, uri=wav_path.stem)

    return {
        'segments': segments,
        'rttm': str(rttm_path),
        'logits': logits,
        'speakers': int(activity.any(axis=0).sum()),
        'frames': int(len(features)),
        'emitted_frames': int(len(logits)),
        'duration_sec': duration,
        'latency_ms_total': elapsed_ms,
        'latency_ms_per_frame': elapsed_ms / max(len(features), 1),
        'rtf': (elapsed_ms / 1000.0) / duration if duration else None,
        'backend': runner.backend,
    }
