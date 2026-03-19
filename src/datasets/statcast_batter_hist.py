"""打者履歴エンコーダ対応 Statcast データセット."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import DataConfig


class StatcastBatterHistDataset(Dataset):
    """打者履歴 + 打席内投球系列対応 Statcast データセット.

    StatcastSequenceDataset の機能に加え、各サンプルに対して
    「その試合の前の直近N打席の全投球データ」を提供する。

    同一 (batter, game_pk) の全投球は同じ打者履歴を共有する。
    """

    def __init__(
        self,
        df: pd.DataFrame,
        cfg: DataConfig,
        max_seq_len: int = 10,
        batter_hist_max_atbats: int = 50,
        batter_hist_max_pitches: int = 10,
        norm_stats: dict[str, tuple[float, float]] | None = None,
        reg_norm_stats: dict[str, tuple[float, float]] | None = None,
        batter_history_dir: Path | None = None,
    ):
        self.cfg = cfg
        self.max_seq_len = max_seq_len
        self.batter_hist_max_atbats = batter_hist_max_atbats
        self.batter_hist_max_pitches = batter_hist_max_pitches
        self.norm_stats = norm_stats or {}
        self.reg_norm_stats = reg_norm_stats or {}

        # === at_bat_id によるグルーピング（投球順を保持） ===
        at_bat_ids = df["at_bat_id"].to_numpy()
        self.atbat_groups: list[np.ndarray] = []
        self.sample_to_group: np.ndarray = np.empty((len(df), 2), dtype=np.int64)

        current_id = at_bat_ids[0]
        group_start = 0
        group_idx = 0
        for row_idx in range(1, len(df)):
            if at_bat_ids[row_idx] != current_id:
                indices = np.arange(group_start, row_idx)
                self.atbat_groups.append(indices)
                for pos, ri in enumerate(indices):
                    self.sample_to_group[ri] = (group_idx, pos)
                group_idx += 1
                group_start = row_idx
                current_id = at_bat_ids[row_idx]
        indices = np.arange(group_start, len(df))
        self.atbat_groups.append(indices)
        for pos, ri in enumerate(indices):
            self.sample_to_group[ri] = (group_idx, pos)

        # === カテゴリカル特徴量 ===
        self.cat_features = {}
        for col in cfg.categorical_features:
            vals = df[col].to_numpy(dtype=np.int64, na_value=-1)
            self.cat_features[col] = torch.from_numpy(vals)

        # === 連続値特徴量（正規化） ===
        cont_arrays = []
        for col in cfg.continuous_features:
            vals = df[col].to_numpy(dtype=np.float32)
            if col in self.norm_stats:
                mean, std = self.norm_stats[col]
                vals = (vals - mean) / std
            vals = np.nan_to_num(vals, nan=0.0)
            cont_arrays.append(vals)
        self.cont_features = torch.from_numpy(
            np.column_stack(cont_arrays) if cont_arrays else np.empty((len(df), 0), dtype=np.float32)
        )
        self.num_cont = self.cont_features.shape[1]

        # === 順序特徴量（float にキャスト） ===
        ord_arrays = []
        for col in cfg.ordinal_features:
            vals = df[col].to_numpy(dtype=np.float32)
            ord_arrays.append(vals)
        self.ord_features = torch.from_numpy(
            np.column_stack(ord_arrays) if ord_arrays else np.empty((len(df), 0), dtype=np.float32)
        )

        # === ターゲット ===
        self.swing_attempt = torch.from_numpy(df[cfg.target_cls_swing_attempt].to_numpy(dtype=np.float32))
        sr = df[cfg.target_cls_swing_result].to_numpy(dtype=np.int64, na_value=-1)
        self.swing_result = torch.from_numpy(sr)
        bt = df[cfg.target_cls_bb_type].to_numpy(dtype=np.int64, na_value=-1)
        self.bb_type = torch.from_numpy(bt)

        reg_arrays = []
        reg_mask_arrays = []
        for col in cfg.target_reg:
            vals = df[col].to_numpy(dtype=np.float32)
            mask = ~np.isnan(vals)
            if col in self.reg_norm_stats:
                mean, std = self.reg_norm_stats[col]
                vals = np.where(mask, (vals - mean) / std, 0.0)
            else:
                vals = np.nan_to_num(vals, nan=0.0)
            reg_arrays.append(vals)
            reg_mask_arrays.append(mask.astype(np.float32))

        self.reg_targets = torch.from_numpy(np.column_stack(reg_arrays))
        self.reg_mask = torch.from_numpy(np.column_stack(reg_mask_arrays))

        # === 打者履歴のセットアップ ===
        self._setup_batter_history(df, batter_history_dir)

    def _setup_batter_history(self, df: pd.DataFrame, batter_history_dir: Path | None) -> None:
        """打者履歴テーブルをロードし、高速ルックアップ用のインデックスを構築する."""
        if batter_history_dir is None:
            batter_history_dir = self.cfg.batter_history_dir

        # (batter, game_pk) → 直近50打席の at_bat_id リスト
        hist_df = pd.read_parquet(batter_history_dir / "batter_game_history.parquet")

        # # at_bat_id → 行インデックスのマッピング (このDatasetのdf内でのインデックス)
        # # まず全データの at_bat_id → グローバル行インデックスをロード
        # global_idx_df = pd.read_parquet(batter_history_dir / "atbat_row_indices.parquet")

        # # このDatasetに含まれる at_bat_id のセットを取得
        # at_bat_ids_in_df = set(df["at_bat_id"].unique())

        # グローバル行インデックスから、このdf内の at_bat_id に対応するものだけ使う
        # ただし行インデックスはこのdfのローカルインデックスではなく
        # 全データの行インデックスなので、dfのat_bat_id→ローカル行インデックスを再構築
        self.atbat_to_local_rows: dict[int, np.ndarray] = {}
        at_bat_id_arr = df["at_bat_id"].to_numpy()
        current_id = at_bat_id_arr[0]
        start = 0
        for i in range(1, len(at_bat_id_arr)):
            if at_bat_id_arr[i] != current_id:
                self.atbat_to_local_rows[int(current_id)] = np.arange(start, i)
                current_id = at_bat_id_arr[i]
                start = i
        self.atbat_to_local_rows[int(current_id)] = np.arange(start, len(at_bat_id_arr))

        # (batter, game_pk) → hist_at_bat_ids のマッピング
        # 履歴の at_bat_id がこのdf内に存在するもののみ有効
        self.batter_game_to_hist: dict[tuple[int, int], list[int]] = {}
        for _, row in hist_df.iterrows():
            batter = int(row["batter"])
            game_pk = int(row["game_pk"])
            hist_ids = row["hist_at_bat_ids"]
            # このdf内に存在する at_bat_id のみフィルタ
            valid_ids = [aid for aid in hist_ids if aid in self.atbat_to_local_rows]
            self.batter_game_to_hist[(batter, game_pk)] = valid_ids

        # 各サンプルの (batter, game_pk) を保持
        self.sample_batter = df["batter"].to_numpy(dtype=np.int64)
        self.sample_game_pk = df["game_pk"].to_numpy(dtype=np.int64)

    def __len__(self) -> int:
        return len(self.swing_attempt)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        group_idx, pos = int(self.sample_to_group[idx, 0]), int(self.sample_to_group[idx, 1])
        group = self.atbat_groups[group_idx]

        # === 打席内投球シーケンス（既存機能） ===
        past_indices = group[:pos]
        if len(past_indices) > self.max_seq_len:
            past_indices = past_indices[-self.max_seq_len :]
        seq_len = len(past_indices)
        T = self.max_seq_len

        seq_pitch_type = torch.zeros(T, dtype=torch.long)
        seq_cont = torch.zeros(T, self.num_cont, dtype=torch.float32)
        seq_swing_attempt = torch.zeros(T, dtype=torch.float32)
        seq_swing_result = torch.full((T,), -1, dtype=torch.long)
        seq_mask = torch.zeros(T, dtype=torch.float32)

        if seq_len > 0:
            pi = torch.from_numpy(past_indices.copy())
            seq_pitch_type[:seq_len] = self.cat_features["pitch_type"][pi]
            seq_cont[:seq_len] = self.cont_features[pi]
            seq_swing_attempt[:seq_len] = self.swing_attempt[pi]
            seq_swing_result[:seq_len] = self.swing_result[pi]
            seq_mask[:seq_len] = 1.0

        # === 打者履歴 ===
        N = self.batter_hist_max_atbats
        P = self.batter_hist_max_pitches

        hist_pitch_type = torch.zeros(N, P, dtype=torch.long)
        hist_cont = torch.zeros(N, P, self.num_cont, dtype=torch.float32)
        hist_swing_attempt = torch.zeros(N, P, dtype=torch.float32)
        hist_swing_result = torch.full((N, P), -1, dtype=torch.long)
        hist_bb_type = torch.full((N,), -1, dtype=torch.long)
        hist_launch_speed = torch.zeros(N, dtype=torch.float32)
        hist_launch_angle = torch.zeros(N, dtype=torch.float32)
        hist_pitch_mask = torch.zeros(N, P, dtype=torch.float32)
        hist_atbat_mask = torch.zeros(N, dtype=torch.float32)

        batter = int(self.sample_batter[idx])
        game_pk = int(self.sample_game_pk[idx])
        hist_at_bat_ids = self.batter_game_to_hist.get((batter, game_pk), [])

        # 直近 N 打席のみ使用（末尾が最新）
        if len(hist_at_bat_ids) > N:
            hist_at_bat_ids = hist_at_bat_ids[-N:]

        for ab_idx, at_bat_id in enumerate(hist_at_bat_ids):
            rows = self.atbat_to_local_rows.get(at_bat_id)
            if rows is None or len(rows) == 0:
                continue

            hist_atbat_mask[ab_idx] = 1.0

            # 投球数を制限
            pitch_rows = rows[:P] if len(rows) > P else rows
            n_pitches = len(pitch_rows)
            ri = torch.from_numpy(pitch_rows)

            hist_pitch_type[ab_idx, :n_pitches] = self.cat_features["pitch_type"][ri]
            hist_cont[ab_idx, :n_pitches] = self.cont_features[ri]
            hist_swing_attempt[ab_idx, :n_pitches] = self.swing_attempt[ri]
            hist_swing_result[ab_idx, :n_pitches] = self.swing_result[ri]
            hist_pitch_mask[ab_idx, :n_pitches] = 1.0

            # 打席結果は最後の投球のもの
            last_row = rows[-1]
            hist_bb_type[ab_idx] = self.bb_type[last_row]
            hist_launch_speed[ab_idx] = self.reg_targets[last_row, 0]  # launch_speed (normalized)
            hist_launch_angle[ab_idx] = self.reg_targets[last_row, 1]  # launch_angle (normalized)

        # 現在の投球の特徴量
        cat = {col: self.cat_features[col][idx] for col in self.cfg.categorical_features}

        return {
            **cat,
            "cont": self.cont_features[idx],
            "ord": self.ord_features[idx],
            "swing_attempt": self.swing_attempt[idx],
            "swing_result": self.swing_result[idx],
            "bb_type": self.bb_type[idx],
            "reg_targets": self.reg_targets[idx],
            "reg_mask": self.reg_mask[idx],
            # 打席内投球シーケンス
            "seq_pitch_type": seq_pitch_type,
            "seq_cont": seq_cont,
            "seq_swing_attempt": seq_swing_attempt,
            "seq_swing_result": seq_swing_result,
            "seq_mask": seq_mask,
            # 打者履歴
            "hist_pitch_type": hist_pitch_type,
            "hist_cont": hist_cont,
            "hist_swing_attempt": hist_swing_attempt,
            "hist_swing_result": hist_swing_result,
            "hist_bb_type": hist_bb_type,
            "hist_launch_speed": hist_launch_speed,
            "hist_launch_angle": hist_launch_angle,
            "hist_pitch_mask": hist_pitch_mask,
            "hist_atbat_mask": hist_atbat_mask,
        }
