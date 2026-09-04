"""LS-EEND streaming speaker diarization on Axera AX650N.

Typical use:

    from ls_eend_sdk import diarize
    result = diarize('meeting.wav', 'models/streaming_step.axmodel')
    print(result['rttm'], result['speakers'])
"""
from .diarize import diarize
from .feature import extract_features, load_audio, wav_to_features, FRAME_SEC
from .postprocess import to_activity, to_segments, write_rttm
from .session import CONV_DELAY, INPUT_NAMES, OUTPUT_NAMES, StreamingDiarizer

__version__ = '1.0.0'

__all__ = [
    'diarize',
    'StreamingDiarizer',
    'load_audio',
    'extract_features',
    'wav_to_features',
    'to_activity',
    'to_segments',
    'write_rttm',
    'CONV_DELAY',
    'FRAME_SEC',
    'INPUT_NAMES',
    'OUTPUT_NAMES',
    '__version__',
]
