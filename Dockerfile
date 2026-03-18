# GPU環境を基準にしたPyTorch公式イメージ
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel

# インストール時の対話プロンプトを無効化
ENV DEBIAN_FRONTEND=noninteractive

# 必要なシステムパッケージのインストール（必要に応じて追加）
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリの設定
WORKDIR /workspace

# Pythonライブラリのインストール
# （リポジトリに requirements.txt がある前提です）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Linter/Formatter 設定をイメージに含める
COPY pyproject.toml .
