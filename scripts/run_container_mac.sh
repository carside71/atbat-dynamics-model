#!/bin/bash

# ==========================================
# 1. 設定項目
# ==========================================
IMAGE_NAME="atbat-dynamics-model-image:latest"
CONTAINER_NAME="atbat-dynamics-model-container"

# マウントするホスト側のディレクトリパス
SRC_DIR="/Users/a81906/my_codes/github/atbat-dynamics-model"
DATA_DIR="/Users/a81906/my_codes/datasets"
OUT_DIR="/Users/a81906/my_codes/experiments/atbat-dynamics-model/00-main"

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
# 3. VS Code 拡張機能の自動インストール設定
# ==========================================
# Attach to Running Container 時に自動インストールされる拡張機能を指定
DEVCONTAINER_METADATA='[{"customizations":{"vscode":{"extensions":["charliermarsh.ruff"],"settings":{"[python]":{"editor.defaultFormatter":"charliermarsh.ruff","editor.formatOnSave":true,"editor.codeActionsOnSave":{"source.fixAll.ruff":"explicit","source.organizeImports.ruff":"explicit"}}}}}}]'

# ==========================================
# 4. コンテナの起動コマンド (バックグラウンド起動)
# ==========================================
echo "🚀 コンテナ [$CONTAINER_NAME] をバックグラウンドで起動しています..."

# -it を外し、-d (バックグラウンド実行) を追加
docker run -d --rm \
  --name "$CONTAINER_NAME" \
  $GPU_FLAG \
  --label "devcontainer.metadata=$DEVCONTAINER_METADATA" \
  -v "$SRC_DIR:/workspace/atbat-dynamics-model" \
  -v "$DATA_DIR:/workspace/datasets" \
  -v "$OUT_DIR:/workspace/outputs" \
  "$IMAGE_NAME" \
  tail -f /dev/null

echo "✅ 起動完了。コンテナに入るには以下のコマンドを実行してください："
echo "   docker exec -it $CONTAINER_NAME bash"