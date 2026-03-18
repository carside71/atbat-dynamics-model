"""モデルレジストリ.

新しいモデルを追加するには:
1. src/models/ にモデルファイルを作成
2. @register_model("name") デコレータでクラスを登録
3. このファイル末尾の import に追加
"""

from __future__ import annotations

import torch.nn as nn

from config import ModelConfig

# ---------------------------------------------------------------------------
# レジストリ
# ---------------------------------------------------------------------------
_MODEL_REGISTRY: dict[str, type[nn.Module]] = {}


def register_model(name: str):
    """モデルクラスをレジストリに登録するデコレータ."""

    def wrapper(cls: type[nn.Module]):
        if name in _MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' is already registered")
        _MODEL_REGISTRY[name] = cls
        return cls

    return wrapper


def create_model(name: str, cfg: ModelConfig, num_cont: int, num_ord: int) -> nn.Module:
    """レジストリからモデルを名前で生成する."""
    if name not in _MODEL_REGISTRY:
        available = ", ".join(sorted(_MODEL_REGISTRY))
        raise ValueError(f"Unknown model '{name}'. Available: {available}")
    return _MODEL_REGISTRY[name](cfg, num_cont, num_ord)


def get_available_models() -> list[str]:
    """登録済みモデル名の一覧を返す."""
    return sorted(_MODEL_REGISTRY)


# ---------------------------------------------------------------------------
# モデルモジュールのインポート（登録を実行するため）
# 新しいモデルを追加したらここに import を追加する
# ---------------------------------------------------------------------------
from models import atbat_dnn  # noqa: E402, F401
