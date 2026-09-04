"""Streaming inference loop for the LS-EEND one-frame AXMODEL.

The graph is a single frame step with explicit recurrent state. Two things are
the caller's responsibility, and getting either wrong costs tens of DER points:

1. **FP32 host accumulation.** The graph returns the *bounded per-frame
   increment* ``k_t v_t`` (``enc{i}_inc`` / ``dec{i}_inc``), not the updated
   retention state. The host keeps the running mean in FP32 via
   ``mean += (inc - mean) / t``. Feeding a quantized state back instead lets
   quantization error compound through the recurrence (DER 1.95% -> 7.4%).

2. **Warmup gating.** Upstream ``StreamingConv1d`` buffers 19 frames and emits
   nothing until its 10th call, so the native loop never invokes the decoder for
   the first ``CONV_DELAY`` frames. Real features are still fed (the encoder
   state must advance), but ``pred`` is discarded and the decoder state is left
   untouched. Skipping this diverges the trajectory permanently (DER -> 71.8%).
"""
from __future__ import annotations

import numpy as np

N_ENC_LAYERS = 4
N_DEC_LAYERS = 2
CONV_DELAY = 9  # StreamingConv1d emits from its (center+1)=10th call
# Output channel count is read from the model at load time: each LS-EEND
# release has a different max_speakers (simu 8 -> 10 channels, AMI 4 -> 6,
# CALLHOME 7 -> 9, DIHARD 10 -> 12).

INPUT_NAMES = (
    ['feat', 'inv_count', 'dec_inv_count', 'conv_cache']
    + [f'enc{i}_{x}' for i in range(N_ENC_LAYERS) for x in ('kv', 'conv')]
    + [f'dec{i}_kv' for i in range(N_DEC_LAYERS)]
)
OUTPUT_NAMES = (
    ['pred']
    + [f'enc{i}_{x}' for i in range(N_ENC_LAYERS) for x in ('inc', 'conv_out')]
    + ['conv_cache_out']
    + [f'dec{i}_inc' for i in range(N_DEC_LAYERS)]
)
ENC_MEAN_NAMES = [f'enc{i}_kv' for i in range(N_ENC_LAYERS)]
DEC_MEAN_NAMES = [f'dec{i}_kv' for i in range(N_DEC_LAYERS)]


class StreamingDiarizer:
    """Frame-synchronous LS-EEND runner over an AXMODEL (or ONNX) session."""

    def __init__(self, model_path, providers=None):
        self.model_path = str(model_path)
        self._session, self._backend = self._open(self.model_path, providers)
        shapes = {i.name: tuple(i.shape) for i in self._session.get_inputs()}
        missing = [n for n in INPUT_NAMES if n not in shapes]
        if missing:
            raise RuntimeError(f'model is missing expected inputs: {missing}')
        self.shapes = shapes
        self.slots = int(np.prod([d for d in self._session.get_outputs()[0].shape]))
        self.reset()

    @staticmethod
    def _open(model_path, providers):
        if model_path.endswith('.axmodel'):
            import axengine as axe
            return axe.InferenceSession(
                model_path, providers=providers or ['AxEngineExecutionProvider']
            ), 'axengine'
        import onnxruntime as ort
        return ort.InferenceSession(
            model_path, providers=providers or ['CPUExecutionProvider']
        ), 'onnxruntime'

    @property
    def backend(self):
        return self._backend

    def reset(self):
        """Clear all recurrent state; call before each new recording."""
        self._state = {n: np.zeros(self.shapes[n], dtype=np.float32) for n in INPUT_NAMES}
        self._enc_t = 0  # frames seen by the encoder
        self._dec_t = 0  # frames actually decoded

    def step(self, frame):
        """Advance one feature frame.

        Args:
            frame: (345,) or (1,1,345) float32 log-mel frame.
        Returns:
            (slots,) float32 logits, or None during the conv warmup.
        """
        state = self._state
        state['feat'] = np.asarray(frame, dtype=np.float32).reshape(self.shapes['feat'])
        enc_b = 1.0 / (self._enc_t + 1)
        dec_b = 1.0 / max(self._dec_t + 1, 1)
        state['inv_count'] = np.full(self.shapes['inv_count'], enc_b, dtype=np.float32)
        state['dec_inv_count'] = np.full(self.shapes['dec_inv_count'], dec_b, dtype=np.float32)

        feed = {n: np.ascontiguousarray(state[n], dtype=np.float32) for n in INPUT_NAMES}
        out = dict(zip(OUTPUT_NAMES, self._session.run(None, feed)))

        # Encoder state always advances. In-place to avoid per-frame allocations.
        for i, name in enumerate(ENC_MEAN_NAMES):
            mean = state[name]
            mean *= (1.0 - enc_b)
            mean += out[f'enc{i}_inc'] * enc_b
            state[f'enc{i}_conv'] = out[f'enc{i}_conv_out']
        state['conv_cache'] = out['conv_cache_out']
        self._enc_t += 1

        if self._enc_t <= CONV_DELAY:
            return None  # conv has not emitted yet; decoder stays frozen

        for i, name in enumerate(DEC_MEAN_NAMES):
            mean = state[name]
            mean *= (1.0 - dec_b)
            mean += out[f'dec{i}_inc'] * dec_b
        self._dec_t += 1
        return np.asarray(out['pred'], dtype=np.float32).reshape(-1)

    def run(self, features, progress=None):
        """Run a whole recording.

        Returns (T - CONV_DELAY, slots) float32 logits. The first
        CONV_DELAY frames produce no output, and the trailing 0.9 s is not
        emitted (the native flush pushes zero *embeddings* past the encoder,
        which a fused one-frame graph cannot express).
        """
        self.reset()
        logits = []
        total = len(features)
        for t in range(total):
            pred = self.step(features[t])
            if pred is not None:
                logits.append(pred)
            if progress is not None and (t + 1) % progress == 0:
                print(f'  {t + 1}/{total} frames', flush=True)
        if not logits:
            raise RuntimeError(f'recording too short: need > {CONV_DELAY} frames, got {total}')
        return np.stack(logits, axis=0)
