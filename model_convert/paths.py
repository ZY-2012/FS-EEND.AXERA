"""Shared paths for the model-conversion scripts.

Everything is resolved relative to this directory, with environment-variable
overrides so nothing is tied to one machine:

    LS_EEND_REPO   upstream FS-EEND checkout's LS-EEND directory
                   (default: ../../FS-EEND/LS-EEND)
    LS_EEND_CKPT   PyTorch checkpoint
                   (default: $LS_EEND_REPO/../ls_eend_1-8spk_16_25_avg_model.ckpt)
    LS_EEND_WAV    reference recording used for calibration and validation
                   (default: $LS_EEND_REPO/test_samples/mix_0000176.wav)
    LS_EEND_OUT    output root for export/ calib_data/ compile/
                   (default: this directory)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _env_path(name, default):
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default


LS = _env_path('LS_EEND_REPO', (HERE / '../../FS-EEND/LS-EEND').resolve())
OUT = _env_path('LS_EEND_OUT', HERE)
CKPT = _env_path('LS_EEND_CKPT', LS.parent / 'ls_eend_1-8spk_16_25_avg_model.ckpt')
WAV = _env_path('LS_EEND_WAV', LS / 'test_samples' / 'mix_0000176.wav')
CONFIG_YAML = LS / 'conf' / 'spk_onl_conformer_retention_enc_dec_nonautoreg_infer.yaml'

EXPORT_DIR = OUT / 'export'
CALIB_DIR = OUT / 'calib_data'
COMPILE_DIR = OUT / 'compile'
ONNX_PATH = EXPORT_DIR / 'streaming_step.onnx'
AXMODEL_PATH = COMPILE_DIR / 'streaming_step.axmodel'


def require(path, hint):
    if not path.exists():
        raise SystemExit(f'missing {path}\n  {hint}')
    return path


def add_upstream_to_syspath():
    """Make the upstream LS-EEND packages importable."""
    require(LS, 'set LS_EEND_REPO to your FS-EEND checkout, e.g. '
                'LS_EEND_REPO=/path/to/FS-EEND/LS-EEND')
    for entry in (str(HERE), str(LS)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
