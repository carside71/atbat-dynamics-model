"""model_scope 関連のテスト."""

import copy
import json
import tempfile
from pathlib import Path

import pytest
import torch

from config import ModelConfig, TrainConfig, validate_model_scope
from losses.multi_task import compute_loss
from models.composable import ComposableModel
from models.components.head_strategies import (
    CascadeHeadStrategy,
    IndependentHeadStrategy,
)


# ---------------------------------------------------------------------------
# バリデーション
# ---------------------------------------------------------------------------


class TestValidateModelScope:
    def test_valid_scopes(self):
        for scope in ("all", "swing_attempt", "outcome", "classification", "regression"):
            validate_model_scope(scope)  # should not raise

    def test_invalid_scope(self):
        with pytest.raises(ValueError, match="Invalid model_scope"):
            validate_model_scope("invalid")


# ---------------------------------------------------------------------------
# IndependentHeadStrategy
# ---------------------------------------------------------------------------


class TestIndependentHeadStrategy:
    BACKBONE_OUT = 16

    def _make_cfg(self, scope: str) -> ModelConfig:
        return ModelConfig(
            model_scope=scope,
            head_hidden=[8],
            head_activation="relu",
            dropout=0.0,
            num_swing_result=3,
            num_bb_type=4,
            num_reg_targets=5,
            regression_head_type="mlp",
        )

    def test_scope_all(self):
        cfg = self._make_cfg("all")
        strategy = IndependentHeadStrategy(cfg, self.BACKBONE_OUT)
        h = torch.randn(4, self.BACKBONE_OUT)
        out = strategy(h)

        assert set(out.keys()) == {"swing_attempt", "swing_result", "bb_type", "regression"}
        assert out["swing_attempt"].shape == (4,)
        assert out["swing_result"].shape == (4, 3)
        assert out["bb_type"].shape == (4, 4)
        assert out["regression"].shape == (4, 5)

    def test_scope_swing_attempt(self):
        cfg = self._make_cfg("swing_attempt")
        strategy = IndependentHeadStrategy(cfg, self.BACKBONE_OUT)
        h = torch.randn(4, self.BACKBONE_OUT)
        out = strategy(h)

        assert set(out.keys()) == {"swing_attempt"}
        assert out["swing_attempt"].shape == (4,)
        assert not hasattr(strategy, "head_swing_result")
        assert not hasattr(strategy, "head_bb_type")
        assert not hasattr(strategy, "head_regression")

    def test_scope_outcome(self):
        cfg = self._make_cfg("outcome")
        strategy = IndependentHeadStrategy(cfg, self.BACKBONE_OUT)
        h = torch.randn(4, self.BACKBONE_OUT)
        out = strategy(h)

        assert set(out.keys()) == {"swing_result", "bb_type", "regression"}
        assert out["swing_result"].shape == (4, 3)
        assert not hasattr(strategy, "head_swing_attempt")

    def test_scope_classification(self):
        cfg = self._make_cfg("classification")
        strategy = IndependentHeadStrategy(cfg, self.BACKBONE_OUT)
        h = torch.randn(4, self.BACKBONE_OUT)
        out = strategy(h)

        assert set(out.keys()) == {"swing_attempt", "swing_result", "bb_type"}
        assert out["swing_attempt"].shape == (4,)
        assert out["swing_result"].shape == (4, 3)
        assert out["bb_type"].shape == (4, 4)
        assert not hasattr(strategy, "head_regression")

    def test_scope_regression(self):
        cfg = self._make_cfg("regression")
        strategy = IndependentHeadStrategy(cfg, self.BACKBONE_OUT)
        h = torch.randn(4, self.BACKBONE_OUT)
        out = strategy(h)

        assert set(out.keys()) == {"regression"}
        assert out["regression"].shape == (4, 5)
        assert not hasattr(strategy, "head_swing_attempt")
        assert not hasattr(strategy, "head_swing_result")
        assert not hasattr(strategy, "head_bb_type")


# ---------------------------------------------------------------------------
# CascadeHeadStrategy
# ---------------------------------------------------------------------------


class TestCascadeHeadStrategy:
    BACKBONE_OUT = 16

    def _make_cfg(self, scope: str) -> ModelConfig:
        return ModelConfig(
            model_scope=scope,
            head_strategy="cascade",
            head_hidden=[8],
            head_activation="relu",
            dropout=0.0,
            detach_cascade=True,
            num_swing_result=3,
            num_bb_type=4,
            num_reg_targets=5,
            regression_head_type="mlp",
        )

    def test_scope_all(self):
        cfg = self._make_cfg("all")
        strategy = CascadeHeadStrategy(cfg, self.BACKBONE_OUT)
        h = torch.randn(4, self.BACKBONE_OUT)
        out = strategy(h)

        assert set(out.keys()) == {"swing_attempt", "swing_result", "bb_type", "regression"}
        assert out["swing_attempt"].shape == (4,)
        assert out["swing_result"].shape == (4, 3)
        assert out["bb_type"].shape == (4, 4)
        assert out["regression"].shape == (4, 5)

    def test_scope_swing_attempt(self):
        cfg = self._make_cfg("swing_attempt")
        strategy = CascadeHeadStrategy(cfg, self.BACKBONE_OUT)
        h = torch.randn(4, self.BACKBONE_OUT)
        out = strategy(h)

        assert set(out.keys()) == {"swing_attempt"}

    def test_scope_outcome(self):
        cfg = self._make_cfg("outcome")
        strategy = CascadeHeadStrategy(cfg, self.BACKBONE_OUT)
        h = torch.randn(4, self.BACKBONE_OUT)
        out = strategy(h)

        assert set(out.keys()) == {"swing_result", "bb_type", "regression"}
        assert out["swing_result"].shape == (4, 3)
        assert out["bb_type"].shape == (4, 4)
        assert out["regression"].shape == (4, 5)

    def test_scope_classification(self):
        cfg = self._make_cfg("classification")
        strategy = CascadeHeadStrategy(cfg, self.BACKBONE_OUT)
        h = torch.randn(4, self.BACKBONE_OUT)
        out = strategy(h)

        assert set(out.keys()) == {"swing_attempt", "swing_result", "bb_type"}
        assert out["swing_attempt"].shape == (4,)
        assert out["swing_result"].shape == (4, 3)
        assert out["bb_type"].shape == (4, 4)
        assert not hasattr(strategy, "head_regression")

    def test_scope_regression(self):
        cfg = self._make_cfg("regression")
        strategy = CascadeHeadStrategy(cfg, self.BACKBONE_OUT)
        h = torch.randn(4, self.BACKBONE_OUT)
        out = strategy(h)

        assert set(out.keys()) == {"regression"}
        assert out["regression"].shape == (4, 5)
        assert not hasattr(strategy, "head_swing_attempt")
        assert not hasattr(strategy, "head_swing_result")
        assert not hasattr(strategy, "head_bb_type")


# ---------------------------------------------------------------------------
# ComposableModel (E2E)
# ---------------------------------------------------------------------------


class TestComposableModel:
    NUM_CONT = 15
    NUM_ORD = 4

    def _make_cfg(self, scope: str) -> ModelConfig:
        return ModelConfig(
            model_scope=scope,
            embedding_dims={
                "p_throws": (3, 2),
                "pitch_type": (5, 4),
            },
            backbone_type="dnn",
            backbone_hidden=[16, 8],
            head_hidden=[8],
            head_activation="relu",
            dropout=0.0,
            num_swing_result=3,
            num_bb_type=4,
            num_reg_targets=5,
            regression_head_type="mlp",
        )

    def _make_inputs(self, B: int = 4):
        cat_dict = {
            "p_throws": torch.randint(0, 3, (B,)),
            "pitch_type": torch.randint(0, 5, (B,)),
        }
        cont = torch.randn(B, self.NUM_CONT)
        ord_feat = torch.randn(B, self.NUM_ORD)
        return cat_dict, cont, ord_feat

    def test_scope_all(self):
        cfg = self._make_cfg("all")
        model = ComposableModel(cfg, self.NUM_CONT, self.NUM_ORD)
        out = model(*self._make_inputs())

        assert set(out.keys()) == {"swing_attempt", "swing_result", "bb_type", "regression"}

    def test_scope_swing_attempt(self):
        cfg = self._make_cfg("swing_attempt")
        model = ComposableModel(cfg, self.NUM_CONT, self.NUM_ORD)
        out = model(*self._make_inputs())

        assert set(out.keys()) == {"swing_attempt"}
        assert out["swing_attempt"].shape == (4,)

    def test_scope_outcome(self):
        cfg = self._make_cfg("outcome")
        model = ComposableModel(cfg, self.NUM_CONT, self.NUM_ORD)
        out = model(*self._make_inputs())

        assert set(out.keys()) == {"swing_result", "bb_type", "regression"}
        assert out["swing_result"].shape == (4, 3)
        assert out["bb_type"].shape == (4, 4)
        assert out["regression"].shape == (4, 5)

    def test_scope_classification(self):
        cfg = self._make_cfg("classification")
        model = ComposableModel(cfg, self.NUM_CONT, self.NUM_ORD)
        out = model(*self._make_inputs())

        assert set(out.keys()) == {"swing_attempt", "swing_result", "bb_type"}
        assert out["swing_attempt"].shape == (4,)
        assert out["swing_result"].shape == (4, 3)
        assert out["bb_type"].shape == (4, 4)

    def test_scope_regression(self):
        cfg = self._make_cfg("regression")
        model = ComposableModel(cfg, self.NUM_CONT, self.NUM_ORD)
        out = model(*self._make_inputs())

        assert set(out.keys()) == {"regression"}
        assert out["regression"].shape == (4, 5)

    def test_scope_all_backward(self):
        """全スコープで勾配が流れることを確認."""
        cfg = self._make_cfg("all")
        model = ComposableModel(cfg, self.NUM_CONT, self.NUM_ORD)
        out = model(*self._make_inputs())
        loss = out["swing_attempt"].sum() + out["swing_result"].sum() + out["bb_type"].sum() + out["regression"].sum()
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None

    def test_scope_classification_backward(self):
        """classification スコープで勾配が流れることを確認."""
        cfg = self._make_cfg("classification")
        model = ComposableModel(cfg, self.NUM_CONT, self.NUM_ORD)
        out = model(*self._make_inputs())
        loss = out["swing_attempt"].sum() + out["swing_result"].sum() + out["bb_type"].sum()
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None

    def test_scope_regression_backward(self):
        """regression スコープで勾配が流れることを確認."""
        cfg = self._make_cfg("regression")
        model = ComposableModel(cfg, self.NUM_CONT, self.NUM_ORD)
        out = model(*self._make_inputs())
        loss = out["regression"].sum()
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None


# ---------------------------------------------------------------------------
# compute_loss
# ---------------------------------------------------------------------------


class TestComputeLoss:
    def test_all_keys(self, train_cfg, fake_batch):
        """全キーが存在する場合、既存と同じ動作."""
        B = fake_batch["swing_attempt"].shape[0]
        outputs = {
            "swing_attempt": torch.randn(B, requires_grad=True),
            "swing_result": torch.randn(B, 3, requires_grad=True),
            "bb_type": torch.randn(B, 4, requires_grad=True),
            "regression": torch.randn(B, 5, requires_grad=True),
        }
        total, losses = compute_loss(outputs, fake_batch, train_cfg)

        assert "total" in losses
        assert "swing_attempt" in losses
        assert "swing_result" in losses
        assert "bb_type" in losses
        assert "regression" in losses
        assert total.requires_grad

    def test_swing_attempt_only(self, train_cfg, fake_batch):
        """swing_attempt のみの outputs で正常動作."""
        B = fake_batch["swing_attempt"].shape[0]
        outputs = {"swing_attempt": torch.randn(B, requires_grad=True)}
        total, losses = compute_loss(outputs, fake_batch, train_cfg)

        assert "swing_attempt" in losses
        assert "swing_result" not in losses
        assert "bb_type" not in losses
        assert "regression" not in losses
        assert total.requires_grad

    def test_outcome_only(self, train_cfg, fake_batch):
        """outcome のみの outputs で正常動作."""
        B = fake_batch["swing_attempt"].shape[0]
        outputs = {
            "swing_result": torch.randn(B, 3),
            "bb_type": torch.randn(B, 4),
            "regression": torch.randn(B, 5),
        }
        total, losses = compute_loss(outputs, fake_batch, train_cfg)

        assert "swing_attempt" not in losses
        assert "swing_result" in losses
        assert "bb_type" in losses
        assert "regression" in losses

    def test_classification_only(self, train_cfg, fake_batch):
        """classification の outputs（SA/SR/BT）で正常動作."""
        B = fake_batch["swing_attempt"].shape[0]
        outputs = {
            "swing_attempt": torch.randn(B, requires_grad=True),
            "swing_result": torch.randn(B, 3, requires_grad=True),
            "bb_type": torch.randn(B, 4, requires_grad=True),
        }
        total, losses = compute_loss(outputs, fake_batch, train_cfg)

        assert "swing_attempt" in losses
        assert "swing_result" in losses
        assert "bb_type" in losses
        assert "regression" not in losses
        assert total.requires_grad

    def test_regression_only(self, train_cfg, fake_batch):
        """regression のみの outputs で正常動作."""
        B = fake_batch["swing_attempt"].shape[0]
        outputs = {"regression": torch.randn(B, 5, requires_grad=True)}
        total, losses = compute_loss(outputs, fake_batch, train_cfg)

        assert "swing_attempt" not in losses
        assert "swing_result" not in losses
        assert "bb_type" not in losses
        assert "regression" in losses
        assert total.requires_grad

    def test_all_keys_total_is_weighted_sum(self, train_cfg, fake_batch):
        """total が各損失の重み付き和になっていることを確認."""
        B = fake_batch["swing_attempt"].shape[0]
        outputs = {
            "swing_attempt": torch.randn(B),
            "swing_result": torch.randn(B, 3),
            "bb_type": torch.randn(B, 4),
            "regression": torch.randn(B, 5),
        }
        total, losses = compute_loss(outputs, fake_batch, train_cfg)

        expected = (
            train_cfg.loss_weight_swing_attempt * losses["swing_attempt"]
            + train_cfg.loss_weight_swing_result * losses["swing_result"]
            + train_cfg.loss_weight_bb_type * losses["bb_type"]
            + train_cfg.loss_weight_regression * losses["regression"]
        )
        assert abs(losses["total"] - expected) < 1e-5


# ---------------------------------------------------------------------------
# model_config.json の保存/復元
# ---------------------------------------------------------------------------


class TestModelConfigPersistence:
    def test_save_load_with_scope(self):
        from utils.model_io import load_trained_model, save_model_config
        from config import DataConfig

        model_cfg = ModelConfig(
            model_scope="swing_attempt",
            embedding_dims={"p_throws": (3, 2), "pitch_type": (5, 4)},
            backbone_type="dnn",
            backbone_hidden=[16, 8],
            head_hidden=[8],
            head_activation="relu",
            dropout=0.0,
            num_swing_result=3,
            num_bb_type=4,
            num_reg_targets=5,
        )
        data_cfg = DataConfig(
            categorical_features=["p_throws", "pitch_type"],
            continuous_features=["f1", "f2"],
            ordinal_features=["o1"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            save_model_config(model_cfg, data_cfg, output_dir)

            # JSON を読み込んで model_scope が保存されていることを確認
            with open(output_dir / "model_config.json") as f:
                saved = json.load(f)
            assert saved["model_scope"] == "swing_attempt"

            # モデルの重みを保存して復元できることを確認
            model = ComposableModel(model_cfg, 2, 1)
            torch.save(model.state_dict(), output_dir / "model.pt")

            loaded_model = load_trained_model(
                output_dir / "model.pt",
                output_dir / "model_config.json",
                torch.device("cpu"),
            )
            # 復元したモデルが swing_attempt のみ出力することを確認
            cat_dict = {"p_throws": torch.tensor([0]), "pitch_type": torch.tensor([0])}
            out = loaded_model(cat_dict, torch.randn(1, 2), torch.randn(1, 1))
            assert set(out.keys()) == {"swing_attempt"}

    def test_backward_compat_no_scope(self):
        """model_scope フィールドがない JSON からロードした場合 'all' になる."""
        from utils.model_io import load_trained_model

        model_cfg = ModelConfig(
            model_scope="all",
            embedding_dims={"p_throws": (3, 2), "pitch_type": (5, 4)},
            backbone_type="dnn",
            backbone_hidden=[16, 8],
            head_hidden=[8],
            head_activation="relu",
            dropout=0.0,
            num_swing_result=3,
            num_bb_type=4,
            num_reg_targets=5,
        )

        # model_scope なしの JSON を手動作成
        saved_json = {
            "backbone_type": "dnn",
            "embedding_dims": {"p_throws": [3, 2], "pitch_type": [5, 4]},
            "backbone_hidden": [16, 8],
            "head_hidden": [8],
            "head_activation": "relu",
            "head_strategy": "independent",
            "detach_cascade": True,
            "regression_head_type": "mlp",
            "dropout": 0.0,
            "num_swing_result": 3,
            "num_bb_type": 4,
            "mdn_num_components": 5,
            "num_cont": 2,
            "num_ord": 1,
            "pitch_seq_max_len": 0,
            "pitch_seq_encoder_type": "gru",
            "pitch_seq_hidden_dim": 64,
            "pitch_seq_num_layers": 1,
            "pitch_seq_bidirectional": False,
            "batter_hist_max_atbats": 0,
            "batter_hist_max_pitches": 10,
            "batter_hist_encoder_type": "gru",
            "batter_hist_hidden_dim": 64,
            "batter_hist_num_layers": 1,
            "num_reg_targets": 5,
            # model_scope は意図的に含めない
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "model_config.json"
            with open(config_path, "w") as f:
                json.dump(saved_json, f)

            model = ComposableModel(model_cfg, 2, 1)
            model_path = Path(tmpdir) / "model.pt"
            torch.save(model.state_dict(), model_path)

            loaded_model = load_trained_model(model_path, config_path, torch.device("cpu"))
            cat_dict = {"p_throws": torch.tensor([0]), "pitch_type": torch.tensor([0])}
            out = loaded_model(cat_dict, torch.randn(1, 2), torch.randn(1, 1))
            # デフォルトは "all" なので全キーが出力される
            assert set(out.keys()) == {"swing_attempt", "swing_result", "bb_type", "regression"}


# ---------------------------------------------------------------------------
# MDN ヘッドとの組み合わせ
# ---------------------------------------------------------------------------


class TestMDNWithScope:
    def test_independent_outcome_mdn(self):
        cfg = ModelConfig(
            model_scope="outcome",
            embedding_dims={"p_throws": (3, 2)},
            backbone_type="dnn",
            backbone_hidden=[16, 8],
            head_hidden=[8],
            head_activation="relu",
            dropout=0.0,
            num_swing_result=3,
            num_bb_type=4,
            num_reg_targets=5,
            regression_head_type="mdn",
            mdn_num_components=3,
        )
        model = ComposableModel(cfg, 2, 1)
        cat_dict = {"p_throws": torch.randint(0, 3, (4,))}
        out = model(cat_dict, torch.randn(4, 2), torch.randn(4, 1))

        assert "swing_attempt" not in out
        assert "swing_result" in out
        reg = out["regression"]
        assert isinstance(reg, dict)
        assert set(reg.keys()) == {"pi", "mu", "sigma"}
