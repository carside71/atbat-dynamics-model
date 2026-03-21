"""打者履歴エンコーダ コンポーネント."""

import torch
import torch.nn as nn

from config import ModelConfig


class HierarchicalGRUBatterHistoryEncoder(nn.Module):
    """階層的 GRU による打者履歴エンコーダ.

    Inner GRU: 各打席の投球列 → 打席ベクトル
    Outer GRU: 打席列 → 打者履歴ベクトル
    """

    def __init__(
        self, cfg: ModelConfig, num_cont: int, seq_pitch_type_embed: nn.Embedding, seq_swing_result_embed: nn.Embedding
    ):
        super().__init__()
        self.cfg = cfg
        self.num_cont = num_cont

        # シーケンスエンコーダと embedding を共有
        self.seq_pitch_type_embed = seq_pitch_type_embed
        self.seq_swing_result_embed = seq_swing_result_embed

        pt_embed_dim = cfg.embedding_dims["pitch_type"][1]
        sr_embed_dim = 4
        hist_pitch_input_dim = pt_embed_dim + num_cont + 1 + sr_embed_dim

        self.inner_gru = nn.GRU(
            input_size=hist_pitch_input_dim,
            hidden_size=cfg.batter_hist_hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        bb_embed_dim = 4
        self.bb_type_embed = nn.Embedding(cfg.num_bb_type + 1, bb_embed_dim)
        atbat_vec_dim = cfg.batter_hist_hidden_dim + bb_embed_dim + 4

        self.outer_gru = nn.GRU(
            input_size=atbat_vec_dim,
            hidden_size=cfg.batter_hist_hidden_dim,
            num_layers=cfg.batter_hist_num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.batter_hist_num_layers > 1 else 0.0,
        )
        self._output_dim = cfg.batter_hist_hidden_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(
        self,
        hist_pitch_type: torch.Tensor,
        hist_cont: torch.Tensor,
        hist_swing_attempt: torch.Tensor,
        hist_swing_result: torch.Tensor,
        hist_bb_type: torch.Tensor,
        hist_launch_speed: torch.Tensor,
        hist_launch_angle: torch.Tensor,
        hist_hc_x: torch.Tensor,
        hist_hc_y: torch.Tensor,
        hist_pitch_mask: torch.Tensor,
        hist_atbat_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, N, P = hist_pitch_type.shape
        device = hist_pitch_type.device

        # --- Inner GRU ---
        flat_pt = hist_pitch_type.reshape(B * N, P)
        flat_cont = hist_cont.reshape(B * N, P, -1)
        flat_sa = hist_swing_attempt.reshape(B * N, P)
        flat_sr = hist_swing_result.reshape(B * N, P)
        flat_pmask = hist_pitch_mask.reshape(B * N, P)

        num_pt = self.cfg.embedding_dims["pitch_type"][0]
        pt_emb = self.seq_pitch_type_embed(flat_pt.clamp(0, num_pt))

        sr = flat_sr.clone()
        sr[sr < 0] = self.cfg.num_swing_result
        sr_emb = self.seq_swing_result_embed(sr)

        inner_feats = torch.cat([pt_emb, flat_cont, flat_sa.unsqueeze(-1), sr_emb], dim=-1)

        inner_lengths = flat_pmask.sum(dim=1).long()
        has_pitches = inner_lengths > 0
        inner_out = torch.zeros(B * N, self.cfg.batter_hist_hidden_dim, device=device)

        if has_pitches.any():
            packed = nn.utils.rnn.pack_padded_sequence(
                inner_feats[has_pitches],
                inner_lengths[has_pitches].cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            _, h_n = self.inner_gru(packed)
            inner_out[has_pitches] = h_n[-1]

        inner_out = inner_out.reshape(B, N, self.cfg.batter_hist_hidden_dim)

        # --- 打席結果を結合 ---
        bb = hist_bb_type.clone()
        bb[bb < 0] = self.cfg.num_bb_type
        bb_emb = self.bb_type_embed(bb)

        atbat_vecs = torch.cat(
            [inner_out, bb_emb, hist_launch_speed.unsqueeze(-1), hist_launch_angle.unsqueeze(-1), hist_hc_x.unsqueeze(-1), hist_hc_y.unsqueeze(-1)],
            dim=-1,
        )

        # --- Outer GRU ---
        atbat_lengths = hist_atbat_mask.sum(dim=1).long()
        has_history = atbat_lengths > 0
        hist_out = torch.zeros(B, self._output_dim, device=device)

        if has_history.any():
            packed = nn.utils.rnn.pack_padded_sequence(
                atbat_vecs[has_history],
                atbat_lengths[has_history].cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            _, h_n = self.outer_gru(packed)
            hist_out[has_history] = h_n[-1]

        return hist_out
