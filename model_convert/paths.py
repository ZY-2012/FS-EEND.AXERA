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
    LS_EEND_CONF   infer YAML basename under $LS_EEND_REPO/conf, which fixes
                   `max_speakers` and therefore the model's output channel count
                   (default: spk_onl_conformer_retention_enc_dec_nonautoreg_infer.yaml,
                   i.e. the simulated-data model with max_speakers=8 -> 10 channels)

Each LS-EEND release is trained with a different `max_speakers`, so the exported
graph shape differs per checkpoint:

    checkpoint                        conf                       max_speakers  channels
    ls_eend_1-8spk_16_25_avg_model    ..._infer.yaml                        8        10
    ls_eend_ami_allspk_model          ..._ami_infer.yaml                    4         6
    ls_eend_ch_allspk_model           ..._callhome_infer.yaml               7         9
    ls_eend_dih2/dih3_allspk_model    ..._dihard2/3_infer.yaml             10        12
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
CONF_NAME = os.environ.get('LS_EEND_CONF',
                           'spk_onl_conformer_retention_enc_dec_nonautoreg_infer.yaml')
CONFIG_YAML = LS / 'conf' / CONF_NAME

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
