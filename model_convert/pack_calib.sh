#!/bin/bash
# Pack the per-input calibration sample directories into the tar.gz files that
# pulsar2 expects (one archive per graph input).
set -euo pipefail
cd "$(dirname "$0")"
[ -d calib_data ] || { echo "run generate_real_calibration.py first"; exit 1; }
rm -f calib_data/*.tar.gz
for d in calib_data/*/; do
    name=$(basename "$d")
    tar -czf "calib_data/$name.tar.gz" -C "$d" .
done
echo "packed $(ls calib_data/*.tar.gz | wc -l) calibration archives"
