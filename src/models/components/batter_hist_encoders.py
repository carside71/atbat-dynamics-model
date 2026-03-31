"""打者履歴エンコーダ コンポーネント."""

import torch
import torch.nn as nn

from config import ModelConfig
from utils.registry import make_registry

BATTER_HIST_ENCODER_REGISTRY, register_batter_hist_encoder = make_registry()


class BaseBatterHistEncoder(nn.Module):
    """打者履歴エンコーダの基底クラス.

    Inner GRU: 各打席の投球列 → 打席ベクトル（全サブクラス共通）
    Outer エンコーダ: 打席列 → 打者履歴ベクトル（サブクラスで実装）
    """

    def __init__(
        self,
        cfg: ModelConfig,
        num_cont: int,
        seq_pitch_type_embed: nn.Embedding | None = None,
        seq_swing_result_embed: nn.Embedding | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.num_cont = num_cont

        # シーケンスエンコーダと embedding を共有（なければ自前で作成）
        if seq_pitch_type_embed is not None:
            self.seq_pitch_type_embed = seq_pitch_type_embed
        else:
            pt_num_classes, pt_embed_dim = cfg.embedding_dims["pitch_type"]
            self.seq_pitch_type_embed = nn.Embedding(pt_num_classes + 1, pt_embed_dim, padding_idx=pt_num_classes)

        if seq_swing_result_embed is not None:
            self.seq_swing_result_embed = seq_swing_result_embed
        else:
            sr_embed_dim = 4
            self.seq_swing_result_embed = nn.Embedding(cfg.num_swing_result + 1, sr_embed_dim)

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
        self.atbat_vec_dim = cfg.batter_hist_hidden_dim + bb_embed_dim + 3

    def _encode_inner(
        self,
        hist_pitch_type: torch.Tensor,
        hist_cont: torch.Tensor,
        hist_swing_attempt: torch.Tensor,
        hist_swing_result: torch.Tensor,
        hist_bb_type: torch.Tensor,
        hist_launch_speed: torch.Tensor,
        hist_launch_angle: torch.Tensor,
        hist_spray_angle: torch.Tensor,
        hist_pitch_mask: torch.Tensor,
        hist_atbat_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inner GRU で各打席の投球列をエンコードし、打席ベクトルを構築する.

        Returns:
            atbat_vecs: (B, N, atbat_vec_dim)
            atbat_lengths: (B,)
        """
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
            [inner_out, bb_emb, hist_launch_speed.unsqueeze(-1), hist_launch_angle.unsqueeze(-1), hist_spray_angle.unsqueeze(-1)],
            dim=-1,
        )

        atbat_lengths = hist_atbat_mask.sum(dim=1).long()
        return atbat_vecs, atbat_lengths


@register_batter_hist_encoder("gru")
class GRUBatterHistEncoder(BaseBatterHistEncoder):
    """GRU ベースの打者履歴エンコーダ."""

    def __init__(
        self, cfg: ModelConfig, num_cont: int, seq_pitch_type_embed: nn.Embedding, seq_swing_result_embed: nn.Embedding
    ):
        super().__init__(cfg, num_cont, seq_pitch_type_embed, seq_swing_result_embed)
        self.outer_gru = nn.GRU(
            input_size=self.atbat_vec_dim,
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
        hist_spray_angle: torch.Tensor,
        hist_pitch_mask: torch.Tensor,
        hist_atbat_mask: torch.Tensor,
    ) -> torch.Tensor:
        B = hist_pitch_type.shape[0]
        device = hist_pitch_type.device

        atbat_vecs, atbat_lengths = self._encode_inner(
            hist_pitch_type, hist_cont, hist_swing_attempt, hist_swing_result,
            hist_bb_type, hist_launch_speed, hist_launch_angle, hist_spray_angle,
            hist_pitch_mask, hist_atbat_mask,
        )

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


@register_batter_hist_encoder("transformer")
class TransformerBatterHistEncoder(BaseBatterHistEncoder):
    """Transformer ベースの打者履歴エンコーダ."""

    def __init__(
        self, cfg: ModelConfig, num_cont: int, seq_pitch_type_embed: nn.Embedding, seq_swing_result_embed: nn.Embedding
    ):
        super().__init__(cfg, num_cont, seq_pitch_type_embed, seq_swing_result_embed)
        self.input_proj = nn.Linear(self.atbat_vec_dim, cfg.batter_hist_hidden_dim)
        nhead = max(1, cfg.batter_hist_hidden_dim // 16)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.batter_hist_hidden_dim,
            nhead=nhead,
            dim_feedforward=cfg.batter_hist_hidden_dim * 4,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.outer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.batter_hist_num_layers, enable_nested_tensor=False)
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
        hist_spray_angle: torch.Tensor,
        hist_pitch_mask: torch.Tensor,
        hist_atbat_mask: torch.Tensor,
    ) -> torch.Tensor:
        B = hist_pitch_type.shape[0]
        device = hist_pitch_type.device

        atbat_vecs, atbat_lengths = self._encode_inner(
            hist_pitch_type, hist_cont, hist_swing_attempt, hist_swing_result,
            hist_bb_type, hist_launch_speed, hist_launch_angle, hist_spray_angle,
            hist_pitch_mask, hist_atbat_mask,
        )

        atbat_vecs = self.input_proj(atbat_vecs)

        key_padding_mask = hist_atbat_mask == 0
        all_padding = atbat_lengths == 0

        if all_padding.all():
            return torch.zeros(B, self._output_dim, device=device)

        temp_mask = key_padding_mask.clone()
        temp_mask[all_padding, 0] = False

        encoded = self.outer_encoder(atbat_vecs, src_key_padding_mask=temp_mask)

        mask_expanded = hist_atbat_mask.unsqueeze(-1)
        lengths_clamped = atbat_lengths.unsqueeze(-1).clamp(min=1).float()
        hist_out = (encoded * mask_expanded).sum(dim=1) / lengths_clamped

        hist_out[all_padding] = 0.0
        return hist_out


# 後方互換エイリアス
HierarchicalGRUBatterHistoryEncoder = GRUBatterHistEncoder
