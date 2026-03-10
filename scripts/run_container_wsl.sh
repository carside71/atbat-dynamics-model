#!/bin/bash

# ==========================================
# 1. 設定項目
# ==========================================
IMAGE_NAME="atbat-dynamics-model-image:latest"
CONTAINER_NAME="atbat-dynamics-model-container"

# マウントするホスト側のディレクトリパス
SRC_DIR="$HOME/github/atbat-dynamics-model"
DATA_DIR="$HOME/datasets"
OUT_DIR="$HOME/experiments/atbat-dynamics-model/00-main"

# ==========================================
# 2. GPUの自動判定
# ==========================================
GPU_FLAG=""
if command -v nvidia-smi &> /dev/null; then
    echo "🟢 NVIDIA GPUを検出しました。GPUモードで起動します。"
    GPU_FLAG="--gpus all"
else
    echo "🟡 NVIDIA GPUが検出されませんでした。CPUモードで起動します。"
fi

# ==========================================
# 3. コンテナの起動コマンド (バックグラウンド起動)
# ==========================================
echo "🚀 コンテナ [$CONTAINER_NAME] をバックグラウンドで起動しています..."

# -it を外し、-d (バックグラウンド実行) を追加
docker run -d --rm \
  --name "$CONTAINER_NAME" \
  $GPU_FLAG \
  -v "$SRC_DIR:/workspace/atbat-dynamics-model" \
  -v "$DATA_DIR:/workspace/datasets" \
  -v "$OUT_DIR:/workspace/outputs" \
  "$IMAGE_NAME" \
  tail -f /dev/null

echo "✅ 起動完了。コンテナに入るには以下のコマンドを実行してください："
echo "   docker exec -it $CONTAINER_NAME bash"
