"""投球シーケンスエンコーダ コンポーネント."""

import torch
import torch.nn as nn

from config import ModelConfig

SEQ_ENCODER_REGISTRY: dict[str, type[nn.Module]] = {}


def register_seq_encoder(name: str):
    def wrapper(cls: type[nn.Module]):
        SEQ_ENCODER_REGISTRY[name] = cls
        return cls

    return wrapper


class BaseSeqEncoder(nn.Module):
    """シーケンスエンコーダの基底クラス."""

    def __init__(self, cfg: ModelConfig, num_cont: int):
        super().__init__()
        self.cfg = cfg

        pt_num_classes, pt_embed_dim = cfg.embedding_dims["pitch_type"]
        self.seq_pitch_type_embed = nn.Embedding(pt_num_classes + 1, pt_embed_dim, padding_idx=pt_num_classes)

        sr_embed_dim = 4
        self.seq_swing_result_embed = nn.Embedding(cfg.num_swing_result + 1, sr_embed_dim)

        self.seq_input_dim = pt_embed_dim + num_cont + 1 + sr_embed_dim

    def _build_seq_features(
        self,
        seq_pitch_type: torch.Tensor,
        seq_cont: torch.Tensor,
        seq_swing_attempt: torch.Tensor,
        seq_swing_result: torch.Tensor,
    ) -> torch.Tensor:
        """共通の特徴量構築ロジック."""
        num_pt = self.cfg.embedding_dims["pitch_type"][0]
        pt = seq_pitch_type.clamp(0, num_pt)
        pt_emb = self.seq_pitch_type_embed(pt)

        sr = seq_swing_result.clone()
        sr[sr < 0] = self.cfg.num_swing_result
        sr_emb = self.seq_swing_result_embed(sr)

        return torch.cat([pt_emb, seq_cont, seq_swing_attempt.unsqueeze(-1), sr_emb], dim=-1)


@register_seq_encoder("gru")
class GRUSeqEncoder(BaseSeqEncoder):
    """GRU ベースの投球系列エンコーダ."""

    def __init__(self, cfg: ModelConfig, num_cont: int):
        super().__init__(cfg, num_cont)
        self.encoder = nn.GRU(
            input_size=self.seq_input_dim,
            hidden_size=cfg.seq_hidden_dim,
            num_layers=cfg.seq_num_layers,
            batch_first=True,
            bidirectional=cfg.seq_bidirectional,
            dropout=cfg.dropout if cfg.seq_num_layers > 1 else 0.0,
        )
        self._output_dim = cfg.seq_hidden_dim * (2 if cfg.seq_bidirectional else 1)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(
        self,
        seq_pitch_type: torch.Tensor,
        seq_cont: torch.Tensor,
        seq_swing_attempt: torch.Tensor,
        seq_swing_result: torch.Tensor,
        seq_mask: torch.Tensor,
    ) -> torch.Tensor:
        B = seq_pitch_type.shape[0]
        device = seq_pitch_type.device
        seq_feats = self._build_seq_features(seq_pitch_type, seq_cont, seq_swing_attempt, seq_swing_result)
        seq_lengths = seq_mask.sum(dim=1).long()
        has_seq = seq_lengths > 0
        seq_out = torch.zeros(B, self._output_dim, device=device)

        if not has_seq.any():
            return seq_out

        packed = nn.utils.rnn.pack_padded_sequence(
            seq_feats[has_seq], seq_lengths[has_seq].cpu(), batch_first=True, enforce_sorted=False
        )
        _, h_n = self.encoder(packed)

        # if self.cfg.seq_bidirectional:
        #     h_n = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        # else:
        #     h_n = h_n[-1]
        h_n = torch.cat([h_n[-2], h_n[-1]], dim=-1) if self.cfg.seq_bidirectional else h_n[-1]

        seq_out[has_seq] = h_n
        return seq_out


@register_seq_encoder("transformer")
class TransformerSeqEncoder(BaseSeqEncoder):
    """Transformer ベースの投球系列エンコーダ."""

    def __init__(self, cfg: ModelConfig, num_cont: int):
        super().__init__(cfg, num_cont)
        self.input_proj = nn.Linear(self.seq_input_dim, cfg.seq_hidden_dim)
        nhead = max(1, cfg.seq_hidden_dim // 16)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.seq_hidden_dim,
            nhead=nhead,
            dim_feedforward=cfg.seq_hidden_dim * 4,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.seq_num_layers)
        self._output_dim = cfg.seq_hidden_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(
        self,
        seq_pitch_type: torch.Tensor,
        seq_cont: torch.Tensor,
        seq_swing_attempt: torch.Tensor,
        seq_swing_result: torch.Tensor,
        seq_mask: torch.Tensor,
    ) -> torch.Tensor:
        B = seq_pitch_type.shape[0]
        device = seq_pitch_type.device
        seq_feats = self._build_seq_features(seq_pitch_type, seq_cont, seq_swing_attempt, seq_swing_result)
        seq_feats = self.input_proj(seq_feats)

        seq_lengths = seq_mask.sum(dim=1).long()
        key_padding_mask = seq_mask == 0

        all_padding = seq_lengths == 0
        if all_padding.all():
            return torch.zeros(B, self._output_dim, device=device)

        temp_mask = key_padding_mask.clone()
        temp_mask[all_padding, 0] = False

        encoded = self.encoder(seq_feats, src_key_padding_mask=temp_mask)

        mask_expanded = seq_mask.unsqueeze(-1)
        lengths_clamped = seq_lengths.unsqueeze(-1).clamp(min=1).float()
        seq_out = (encoded * mask_expanded).sum(dim=1) / lengths_clamped

        seq_out[all_padding] = 0.0
        return seq_out
