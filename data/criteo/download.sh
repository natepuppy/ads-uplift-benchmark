#!/usr/bin/env bash
# Download and decompress the Criteo Uplift Modeling Dataset v2.1
#
# Dataset details:
#   - 13.98M rows, 12 features + treatment + conversion + visit labels
#   - Random treatment assignment (A/B test), so no confounding
#   - Used as a real-data sanity check: synthetic findings should generalize
#
# Reference:
#   Diemert et al. (2018). "A Large Scale Benchmark for Uplift Modeling."
#   AdKDD Workshop at KDD 2018.
#   https://ailab.criteo.com/criteo-uplift-prediction-dataset/

set -euo pipefail

DATA_DIR="$(dirname "$0")"
URL="http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz"
COMPRESSED="$DATA_DIR/criteo_uplift_v2.csv.gz"
OUTPUT="$DATA_DIR/criteo_uplift_v2.csv"

if [ -f "$OUTPUT" ]; then
    echo "Criteo dataset already downloaded: $OUTPUT"
    exit 0
fi

echo "Downloading Criteo Uplift Dataset v2.1 (~300MB compressed)..."
curl -L "$URL" -o "$COMPRESSED" --progress-bar

echo "Decompressing..."
gunzip -k "$COMPRESSED"

echo "Done. Dataset saved to: $OUTPUT"
echo "Run 'python data/criteo/preprocess.py' to preprocess."
