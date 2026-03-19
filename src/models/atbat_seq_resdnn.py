"""シーケンスエンコーダ付き残差 DNN モデル.

同一打席内の過去投球系列を GRU / Transformer でエンコードし、
現在の投球特徴量に結合して予測を行う。
"""

import torch
import torch.nn as nn

from config import ModelConfig
from models import register_model
from models.atbat_resdnn import ProjectedResBlock, ResBlock


@register_model("atbat_seq_resdnn")
class AtBatSeqResDNN(nn.Module):
    """シーケンスエンコーダ + 残差接続 + GELU による打席結果予測モデル.

    構造:
      過去投球系列 → Sequence Encoder → seq_embedding
      現在の投球特徴量 + seq_embedding → 共有バックボーン → ヘッド群
    """

    is_seq_model = True

    def __init__(self, cfg: ModelConfig, num_cont: int, num_ord: int):
        super().__init__()
        self.cfg = cfg

        # === 現在の投球の Embedding layers ===
        self.embeddings = nn.ModuleDict()
        embed_total_dim = 0
        for feat_name, (num_classes, embed_dim) in cfg.embedding_dims.items():
            self.embeddings[feat_name] = nn.Embedding(num_classes + 1, embed_dim, padding_idx=num_classes)
            embed_total_dim += embed_dim

        # === シーケンスエンコーダ ===
        pt_num_classes, pt_embed_dim = cfg.embedding_dims["pitch_type"]
        self.seq_pitch_type_embed = nn.Embedding(pt_num_classes + 1, pt_embed_dim, padding_idx=pt_num_classes)

        # swing_result embedding: 0..num_swing_result-1 (有効値) + num_swing_result (no_swing)
        sr_embed_dim = 4
        self.seq_swing_result_embed = nn.Embedding(cfg.num_swing_result + 1, sr_embed_dim)

        # シーケンス入力次元: pitch_type_emb + 連続値特徴量 + swing_attempt + swing_result_emb
        seq_input_dim = pt_embed_dim + num_cont + 1 + sr_embed_dim

        if cfg.seq_encoder_type == "gru":
            self.seq_encoder = nn.GRU(
                input_size=seq_input_dim,
                hidden_size=cfg.seq_hidden_dim,
                num_layers=cfg.seq_num_layers,
                batch_first=True,
                bidirectional=cfg.seq_bidirectional,
                dropout=cfg.dropout if cfg.seq_num_layers > 1 else 0.0,
            )
            seq_out_dim = cfg.seq_hidden_dim * (2 if cfg.seq_bidirectional else 1)
        elif cfg.seq_encoder_type == "transformer":
            self.seq_input_proj = nn.Linear(seq_input_dim, cfg.seq_hidden_dim)
            nhead = max(1, cfg.seq_hidden_dim // 16)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=cfg.seq_hidden_dim,
                nhead=nhead,
                dim_feedforward=cfg.seq_hidden_dim * 4,
                dropout=cfg.dropout,
                batch_first=True,
            )
            self.seq_encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.seq_num_layers)
            seq_out_dim = cfg.seq_hidden_dim
        else:
            raise ValueError(f"Unknown seq_encoder_type: {cfg.seq_encoder_type}")

        self.seq_out_dim = seq_out_dim

        # === 共有バックボーン (ResBlocks) ===
        input_dim = embed_total_dim + num_cont + num_ord + seq_out_dim

        blocks: list[nn.Module] = []
        in_dim = input_dim
        for hidden_dim in cfg.backbone_hidden:
            if in_dim != hidden_dim:
                blocks.append(ProjectedResBlock(in_dim, hidden_dim, cfg.dropout))
            else:
                blocks.append(ResBlock(hidden_dim, cfg.dropout))
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*blocks)
        backbone_out = cfg.backbone_hidden[-1]

        # === ヘッド ===
        self.head_swing_attempt = self._build_head(backbone_out, cfg.head_hidden, 1)
        self.head_swing_result = self._build_head(backbone_out, cfg.head_hidden, cfg.num_swing_result)
        self.head_bb_type = self._build_head(backbone_out, cfg.head_hidden, cfg.num_bb_type)
        self.head_regression = self._build_head(backbone_out, cfg.head_hidden, 3)

    def _build_head(self, in_dim: int, hidden_dims: list[int], out_dim: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden_dims:
            layers.extend(
                [
                    nn.Linear(d, h),
                    nn.GELU(),
                    nn.Dropout(self.cfg.dropout),
                ]
            )
            d = h
        layers.append(nn.Linear(d, out_dim))
        return nn.Sequential(*layers)

    def _encode_sequence(
        self,
        seq_pitch_type: torch.Tensor,
        seq_cont: torch.Tensor,
        seq_swing_attempt: torch.Tensor,
        seq_swing_result: torch.Tensor,
        seq_mask: torch.Tensor,
    ) -> torch.Tensor:
        """過去投球系列をエンコードする.

        Args:
            seq_pitch_type: (B, T) 過去の pitch_type
            seq_cont: (B, T, num_cont) 過去の連続特徴量（正規化済み）
            seq_swing_attempt: (B, T) 過去の swing_attempt
            seq_swing_result: (B, T) 過去の swing_result (-1 = no swing)
            seq_mask: (B, T) 有効マスク (1=有効, 0=パディング)

        Returns:
            (B, seq_out_dim) シーケンス埋め込み
        """
        B, T = seq_pitch_type.shape
        device = seq_pitch_type.device

        # pitch_type embedding
        num_pt = self.cfg.embedding_dims["pitch_type"][0]
        pt = seq_pitch_type.clamp(0, num_pt)
        pt_emb = self.seq_pitch_type_embed(pt)  # (B, T, pt_embed_dim)

        # swing_result embedding: -1 → num_swing_result (no_swing)
        sr = seq_swing_result.clone()
        sr[sr < 0] = self.cfg.num_swing_result
        sr_emb = self.seq_swing_result_embed(sr)  # (B, T, sr_embed_dim)

        # 全シーケンス特徴量を結合
        seq_feats = torch.cat(
            [
                pt_emb,
                seq_cont,
                seq_swing_attempt.unsqueeze(-1),
                sr_emb,
            ],
            dim=-1,
        )  # (B, T, seq_input_dim)

        seq_lengths = seq_mask.sum(dim=1).long()  # (B,)

        if self.cfg.seq_encoder_type == "gru":
            return self._encode_gru(seq_feats, seq_lengths, B, device)
        else:
            return self._encode_transformer(seq_feats, seq_lengths, seq_mask, B, device)

    def _encode_gru(
        self,
        seq_feats: torch.Tensor,
        seq_lengths: torch.Tensor,
        B: int,
        device: torch.device,
    ) -> torch.Tensor:
        has_seq = seq_lengths > 0
        seq_out = torch.zeros(B, self.seq_out_dim, device=device)

        if not has_seq.any():
            return seq_out

        valid_feats = seq_feats[has_seq]
        valid_lengths = seq_lengths[has_seq]

        packed = nn.utils.rnn.pack_padded_sequence(
            valid_feats, valid_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h_n = self.seq_encoder(packed)
        # h_n: (num_layers * num_dirs, N_valid, hidden_dim)

        h_n = torch.cat([h_n[-2], h_n[-1]], dim=-1) if self.cfg.seq_bidirectional else h_n[-1]

        seq_out[has_seq] = h_n
        return seq_out

    def _encode_transformer(
        self,
        seq_feats: torch.Tensor,
        seq_lengths: torch.Tensor,
        seq_mask: torch.Tensor,
        B: int,
        device: torch.device,
    ) -> torch.Tensor:
        seq_feats = self.seq_input_proj(seq_feats)  # (B, T, seq_hidden_dim)

        # key_padding_mask: True = 無視
        key_padding_mask = seq_mask == 0

        # 全パディングのサンプルは最初の位置を有効にして処理（後でゼロ化）
        all_padding = seq_lengths == 0
        if all_padding.all():
            return torch.zeros(B, self.cfg.seq_hidden_dim, device=device)

        temp_mask = key_padding_mask.clone()
        temp_mask[all_padding, 0] = False

        encoded = self.seq_encoder(seq_feats, src_key_padding_mask=temp_mask)

        # 有効位置の平均プーリング
        mask_expanded = seq_mask.unsqueeze(-1)
        lengths_clamped = seq_lengths.unsqueeze(-1).clamp(min=1).float()
        seq_out = (encoded * mask_expanded).sum(dim=1) / lengths_clamped

        # 全パディングのサンプルをゼロ化
        seq_out[all_padding] = 0.0
        return seq_out

    def forward(
        self,
        cat_dict: dict[str, torch.Tensor],
        cont: torch.Tensor,
        ord_feat: torch.Tensor,
        seq_pitch_type: torch.Tensor | None = None,
        seq_cont: torch.Tensor | None = None,
        seq_swing_attempt: torch.Tensor | None = None,
        seq_swing_result: torch.Tensor | None = None,
        seq_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # Embed categorical features
        embeds = []
        for feat_name in self.cfg.embedding_dims:
            x = cat_dict[feat_name]
            num_classes = self.cfg.embedding_dims[feat_name][0]
            x = torch.where((x < 0) | (x >= num_classes), num_classes, x)
            embeds.append(self.embeddings[feat_name](x))

        # シーケンスエンコード
        if seq_pitch_type is not None:
            seq_emb = self._encode_sequence(seq_pitch_type, seq_cont, seq_swing_attempt, seq_swing_result, seq_mask)
        else:
            B = cont.shape[0]
            seq_emb = torch.zeros(B, self.seq_out_dim, device=cont.device)

        # 全入力を結合
        parts = embeds + [cont, ord_feat, seq_emb]
        x = torch.cat(parts, dim=-1)

        # 共有バックボーン
        h = self.backbone(x)

        return {
            "swing_attempt": self.head_swing_attempt(h).squeeze(-1),
            "swing_result": self.head_swing_result(h),
            "bb_type": self.head_bb_type(h),
            "regression": self.head_regression(h),
        }
