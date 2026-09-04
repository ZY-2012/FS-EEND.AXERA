#!/bin/bash
# Quantize + compile the exported streaming graph for AX650/NPU3.
#
# Requires Docker and the Pulsar2 7.0 image. Run from model_convert/ after
# export_streaming_step.py and generate_real_calibration.py.
set -euo pipefail
cd "$(dirname "$0")"

IMAGE="${PULSAR2_IMAGE:-docker-registry.aitsw.axera-tech.com/pulsar2:20260810-temp-09cadfa9}"

[ -f export/streaming_step.onnx ] || { echo "missing export/streaming_step.onnx (run export_streaming_step.py)"; exit 1; }
[ -f calib_data/feat.tar.gz ]     || { echo "missing calib_data/*.tar.gz (run generate_real_calibration.py + pack_calib.sh)"; exit 1; }

docker run --rm -v "$PWD":/workspace -w /workspace "$IMAGE" \
    pulsar2 build --config /workspace/pulsar2_config.json

echo "built: $PWD/compile/streaming_step.axmodel"
