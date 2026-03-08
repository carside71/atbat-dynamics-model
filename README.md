# atbat-dynamics-model

簡単な力学モデル / 実験用リポジトリです。

**目的**: モデルの学習・実験をコンテナまたはローカル環境で実行できるようにする。

**注意**: リポジトリ内の `requirements.txt` は現状ライブラリ未記載です。必要な依存を追記してからインストールしてください。

**Prerequisites**
- **Docker**: コンテナ実行・イメージ作成に使用します。
- **Python**: ローカル実行時は Python 3.8 以上を推奨します。

**ローカル環境（venv）でのセットアップ**

1. 仮想環境の作成・有効化

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. 依存のインストール

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

※ `requirements.txt` にライブラリが無い場合は、必要なパッケージを追加してください。

**Dockerでのセットアップと実行**

1. イメージのビルド（プロジェクトルートで実行）

```bash
docker build -t atbat-dynamics-model-image:latest .
```

2. 実行スクリプトを使ってコンテナ起動

```bash
chmod +x scripts/run_container.sh
./scripts/run_container.sh
```

`scripts/run_container.sh` は環境に合わせて以下のホストパス変数を編集して使用してください：
- `SRC_DIR`（ソースのあるディレクトリ）
- `DATA_DIR`（データセットの格納ディレクトリ）
- `OUT_DIR`（出力ファイル保存先）

GPUについて: スクリプトは `nvidia-smi` の有無でGPUモードを判定します。NVIDIA GPU を使う場合はホストに NVIDIA ドライバと NVIDIA Container Toolkit が必要です。macOSでは通常NVIDIA GPUは利用できないため、CPUモードで起動します。

コンテナに入るには:

```bash
docker exec -it atbat-dynamics-model-container bash
```

**補足**
- Dockerfile は公式 PyTorch イメージ（GPU対応）をベースに `requirements.txt` をインストールする仕様です。
- 依存関係が未記載の場合は、`requirements.txt` に必要パッケージを追加してください。

---

ファイル: [scripts/run_container.sh](scripts/run_container.sh) を参照して、マウント設定や起動方法を確認してください。
