"""8 kHz log-mel frontend for LS-EEND (``logmel23_cummn``).

Mirrors ``FS-EEND/LS-EEND/datasets/feature.py`` exactly: 23 mel bins, cumulative
mean normalization, +-7 frame splicing, then 10x subsampling. One output frame
therefore covers 0.1 s of audio.
"""
from __future__ import annotations

import numpy as np

SAMPLE_RATE = 8000
N_MELS = 23
CONTEXT = 7
SUBSAMPLING = 10
FRAME_SHIFT = 80
WIN_LENGTH = 200
N_FFT = 1024
FEATURE_DIM = (2 * CONTEXT + 1) * N_MELS
FRAME_SEC = FRAME_SHIFT * SUBSAMPLING / SAMPLE_RATE  # 0.1 s


def load_audio(wav_path):
    """Read a wav as mono float32 at 8 kHz. Returns (audio, duration_seconds)."""
    import soundfile as sf

    audio, sr = sf.read(str(wav_path), dtype='float32', always_2d=False)
    if audio.ndim > 1:
        audio = audio[:, 0]
    duration = len(audio) / sr
    if sr != SAMPLE_RATE:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return audio, duration


def extract_features(audio):
    """audio (float32, 8 kHz) -> features (T, 345) float32."""
    import librosa

    spec = librosa.stft(audio, n_fft=N_FFT, win_length=WIN_LENGTH, hop_length=FRAME_SHIFT).T
    if len(audio) % FRAME_SHIFT == 0:
        spec = spec[:-1]
    mag = np.abs(spec)
    mel_fb = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=2 * (mag.shape[1] - 1), n_mels=N_MELS)
    logmel = np.log10(np.maximum(np.dot(mag ** 2, mel_fb.T), 1e-10))

    # Cumulative mean normalization: frame t is normalized by the mean of frames 0..t.
    cum = np.cumsum(logmel, axis=0)
    idx = np.arange(1, logmel.shape[0] + 1, dtype=np.float32)
    logmel = logmel - cum / idx[:, None]

    padded = np.pad(logmel, ((CONTEXT, CONTEXT), (0, 0)), mode='constant')
    n = logmel.shape[0]
    spliced = np.lib.stride_tricks.as_strided(
        padded, (n, FEATURE_DIM), (padded.itemsize * N_MELS, padded.itemsize)
    ).copy()
    return spliced[::SUBSAMPLING].astype(np.float32)


def wav_to_features(wav_path):
    """Convenience wrapper: wav path -> (features (T,345), duration_seconds)."""
    audio, duration = load_audio(wav_path)
    return extract_features(audio), duration
