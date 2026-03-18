# モデルの追加方法

## 概要

このディレクトリにはモデルアーキテクチャの実装を配置します。
レジストリパターンにより、新しいモデルを追加するだけで学習・評価パイプラインから利用できます。

## 手順

### 1. モデルファイルを作成

`src/models/` に Python ファイルを作成します。

```python
# src/models/my_model.py
import torch
import torch.nn as nn

from config import ModelConfig
from models import register_model


@register_model("my_model")
class MyModel(nn.Module):

    def __init__(self, cfg: ModelConfig, num_cont: int, num_ord: int):
        super().__init__()
        # cfg.embedding_dims, cfg.backbone_hidden 等を使ってネットワークを構築
        ...

    def forward(
        self,
        cat_dict: dict[str, torch.Tensor],
        cont: torch.Tensor,
        ord_feat: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        ...
        return {
            "swing_attempt": ...,  # (B,) logits
            "swing_result": ...,   # (B, num_swing_result) logits
            "bb_type": ...,        # (B, num_bb_type) logits
            "regression": ...,     # (B, 3) 予測値
        }
```

### 2. `__init__.py` にインポートを追加

`src/models/__init__.py` 末尾のインポートブロックにモジュールを追加します。

```python
from models import atbat_dnn  # noqa: E402, F401
from models import my_model   # noqa: E402, F401  # ← 追加
```

### 3. 設定ファイル (YAML) で指定

```yaml
model:
  architecture: my_model
  # その他のハイパーパラメータ
```

## 規約

| 項目 | 要件 |
|------|------|
| コンストラクタ引数 | `(cfg: ModelConfig, num_cont: int, num_ord: int)` |
| `forward` 引数 | `(cat_dict, cont, ord_feat)` |
| `forward` 戻り値 | `dict` with keys: `swing_attempt`, `swing_result`, `bb_type`, `regression` |
| 登録名 | `@register_model("名前")` で一意な名前を付ける |

## 登録済みモデル一覧

| 名前 | ファイル | 説明 |
|------|----------|------|
| `atbat_dnn` | `atbat_dnn.py` | 共有バックボーン + 4 ヘッドの MLP |
