"""打者履歴エンコーダ対応 Statcast データセット."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import DataConfig
from datasets.statcast_base import StatcastBaseDataset


class StatcastBatterHistDataset(StatcastBaseDataset):
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
        super().__init__(df, cfg, norm_stats, reg_norm_stats)
        self.max_seq_len = max_seq_len
        self.batter_hist_max_atbats = batter_hist_max_atbats
        self.batter_hist_max_pitches = batter_hist_max_pitches

        self._build_atbat_groups(df["at_bat_id"].to_numpy())

        # === パディング用ダミー行の追加（打者履歴の無効エントリ参照用） ===
        self._n_real_samples = len(df)
        self._dummy_idx = self._n_real_samples
        self.cat_features["pitch_type"] = torch.cat([self.cat_features["pitch_type"], torch.zeros(1, dtype=torch.long)])
        self.cont_features = torch.cat([self.cont_features, torch.zeros(1, self.num_cont)])
        self.swing_attempt = torch.cat([self.swing_attempt, torch.zeros(1)])
        self.swing_result = torch.cat([self.swing_result, torch.full((1,), -1, dtype=torch.long)])
        self.bb_type = torch.cat([self.bb_type, torch.full((1,), -1, dtype=torch.long)])
        self.reg_targets = torch.cat([self.reg_targets, torch.zeros(1, self.reg_targets.shape[1])])

        # === 打者履歴のセットアップ（インデックスを事前計算） ===
        self._setup_batter_history(df, batter_history_dir)

    def _setup_batter_history(self, df: pd.DataFrame, batter_history_dir: Path | None) -> None:
        """打者履歴のルックアップインデックスを事前計算する."""
        if batter_history_dir is None:
            batter_history_dir = self.cfg.batter_history_dir

        N = self.batter_hist_max_atbats
        P = self.batter_hist_max_pitches
        dummy_idx = self._dummy_idx

        # at_bat_id → ローカル行インデックス
        atbat_to_local_rows: dict[int, np.ndarray] = {}
        at_bat_id_arr = df["at_bat_id"].to_numpy()
        current_id = at_bat_id_arr[0]
        start = 0
        for i in range(1, len(at_bat_id_arr)):
            if at_bat_id_arr[i] != current_id:
                atbat_to_local_rows[int(current_id)] = np.arange(start, i)
                current_id = at_bat_id_arr[i]
                start = i
        atbat_to_local_rows[int(current_id)] = np.arange(start, len(at_bat_id_arr))

        # (batter, game_pk) → hist_at_bat_ids のマッピング
        hist_df = pd.read_parquet(batter_history_dir / "batter_game_history.parquet")
        batter_game_to_hist: dict[tuple[int, int], list[int]] = {}
        for _, row in hist_df.iterrows():
            batter = int(row["batter"])
            game_pk = int(row["game_pk"])
            hist_ids = row["hist_at_bat_ids"]
            valid_ids = [aid for aid in hist_ids if aid in atbat_to_local_rows]
            if valid_ids:
                batter_game_to_hist[(batter, game_pk)] = valid_ids

        # ユニークな (batter, game_pk) ペアを特定し、各サンプルをマッピング
        sample_batter = df["batter"].to_numpy(dtype=np.int64)
        sample_game_pk = df["game_pk"].to_numpy(dtype=np.int64)
        bg_keys = np.column_stack([sample_batter, sample_game_pk])
        unique_pairs, bg_inverse = np.unique(bg_keys, axis=0, return_inverse=True)
        self._sample_bg_idx = bg_inverse.astype(np.int64)

        U = len(unique_pairs)
        bg_pair_to_idx = {(int(row[0]), int(row[1])): idx for idx, row in enumerate(unique_pairs)}

        # 事前計算配列の初期化（無効エントリはダミー行を参照）
        self._bg_rows_2d = np.full((U, N, P), dummy_idx, dtype=np.int64)
        self._bg_pitch_mask = np.zeros((U, N, P), dtype=np.bool_)
        self._bg_atbat_mask = np.zeros((U, N), dtype=np.bool_)
        self._bg_last_rows = np.full((U, N), dummy_idx, dtype=np.int64)

        for pair, bg_idx in bg_pair_to_idx.items():
            hist_at_bat_ids = batter_game_to_hist.get(pair, [])
            if len(hist_at_bat_ids) > N:
                hist_at_bat_ids = hist_at_bat_ids[-N:]

            for ab_idx, at_bat_id in enumerate(hist_at_bat_ids):
                rows = atbat_to_local_rows.get(at_bat_id)
                if rows is None or len(rows) == 0:
                    continue

                self._bg_atbat_mask[bg_idx, ab_idx] = True
                pitch_rows = rows[:P] if len(rows) > P else rows
                n_pitches = len(pitch_rows)
                self._bg_rows_2d[bg_idx, ab_idx, :n_pitches] = pitch_rows
                self._bg_pitch_mask[bg_idx, ab_idx, :n_pitches] = True
                self._bg_last_rows[bg_idx, ab_idx] = rows[-1]

        print(f"  Batter history pre-computed: {U} unique (batter, game_pk) pairs")

    def __len__(self) -> int:
        return self._n_real_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        group_idx, pos = int(self.sample_to_group[idx, 0]), int(self.sample_to_group[idx, 1])
        past_indices = self.atbat_groups[group_idx][:pos]

        result = self._base_item(idx)
        result.update(self._build_seq_features(past_indices, self.max_seq_len))

        # === 打者履歴（事前計算済みインデックスによるベクトル化） ===
        N = self.batter_hist_max_atbats
        P = self.batter_hist_max_pitches

        bg_idx = self._sample_bg_idx[idx]

        rows_tensor = torch.from_numpy(self._bg_rows_2d[bg_idx].reshape(-1).copy())  # (N*P,)
        hist_pitch_type = self.cat_features["pitch_type"][rows_tensor].reshape(N, P)
        hist_cont = self.cont_features[rows_tensor].reshape(N, P, -1)
        hist_swing_attempt = self.swing_attempt[rows_tensor].reshape(N, P)
        hist_swing_result = self.swing_result[rows_tensor].reshape(N, P)

        hist_pitch_mask = torch.tensor(self._bg_pitch_mask[bg_idx], dtype=torch.float32)
        hist_atbat_mask = torch.tensor(self._bg_atbat_mask[bg_idx], dtype=torch.float32)

        last_tensor = torch.from_numpy(self._bg_last_rows[bg_idx].copy())  # (N,)
        hist_bb_type = self.bb_type[last_tensor]
        hist_launch_speed = self.reg_targets[last_tensor, 0]
        hist_launch_angle = self.reg_targets[last_tensor, 1]
        hist_spray_angle = self.reg_targets[last_tensor, 3]

        result.update({
            "hist_pitch_type": hist_pitch_type,
            "hist_cont": hist_cont,
            "hist_swing_attempt": hist_swing_attempt,
            "hist_swing_result": hist_swing_result,
            "hist_bb_type": hist_bb_type,
            "hist_launch_speed": hist_launch_speed,
            "hist_launch_angle": hist_launch_angle,
            "hist_spray_angle": hist_spray_angle,
            "hist_pitch_mask": hist_pitch_mask,
            "hist_atbat_mask": hist_atbat_mask,
        })
        return result
