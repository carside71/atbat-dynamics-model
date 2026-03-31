#!/bin/bash
# ヒートマップ回帰モデルの推論結果を可視化する
#
# 使い方:
#   bash scripts/visualize_heatmap.sh <model-dir>
#   bash scripts/visualize_heatmap.sh <model-dir> <num-samples> <split>
#
# 例:
#   bash scripts/visualize_heatmap.sh /workspace/outputs/atbat-dynamics-model/2026-03-28-120000
#   bash scripts/visualize_heatmap.sh /workspace/outputs/atbat-dynamics-model/2026-03-28-120000 20 val

set -euo pipefail

cd /workspace/atbat-dynamics-model

MODEL_DIR="${1:?Usage: $0 <model-dir> [num-samples] [split]}"
NUM_SAMPLES="${2:-10}"
SPLIT="${3:-test}"

python -m tools.visualize_heatmap \
    --model-dir "${MODEL_DIR}" \
    --num-samples "${NUM_SAMPLES}" \
    --split "${SPLIT}" \
    --overview-grid
