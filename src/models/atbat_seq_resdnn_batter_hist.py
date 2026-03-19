"""打者履歴エンコーダ付きシーケンス残差 DNN モデル.

AtBatSeqResDNN を拡張し、打者の直近N打席の全投球データを
階層的 GRU（Inner: 打席内投球→打席ベクトル, Outer: 打席列→打者履歴ベクトル）
でエンコードして予測に利用する。
"""

import torch
import torch.nn as nn

from config import ModelConfig
from models import register_model
from models.atbat_resdnn import ProjectedResBlock, ResBlock


@register_model("atbat_seq_resdnn_batter_hist")
class AtBatSeqResDNNBatterHist(nn.Module):
    """打者履歴エンコーダ + シーケンスエンコーダ + 残差 DNN.

    構造:
      打者直近N打席 → Hierarchical GRU → batter_hist_emb
      打席内過去投球 → Pitch Seq Encoder → seq_emb
      現在の投球特徴量 → Embedding + Concat → feat_vec
      [feat_vec, seq_emb, batter_hist_emb] → Shared Backbone → Heads
    """

    is_seq_model = True
    is_batter_hist_model = True

    def __init__(self, cfg: ModelConfig, num_cont: int, num_ord: int):
        super().__init__()
        self.cfg = cfg
        self.num_cont = num_cont

        # === 現在の投球の Embedding layers ===
        self.embeddings = nn.ModuleDict()
        embed_total_dim = 0
        for feat_name, (num_classes, embed_dim) in cfg.embedding_dims.items():
            self.embeddings[feat_name] = nn.Embedding(num_classes + 1, embed_dim, padding_idx=num_classes)
            embed_total_dim += embed_dim

        # === 打席内投球シーケンスエンコーダ (既存) ===
        pt_num_classes, pt_embed_dim = cfg.embedding_dims["pitch_type"]
        self.seq_pitch_type_embed = nn.Embedding(pt_num_classes + 1, pt_embed_dim, padding_idx=pt_num_classes)

        sr_embed_dim = 4
        self.seq_swing_result_embed = nn.Embedding(cfg.num_swing_result + 1, sr_embed_dim)

        seq_input_dim = pt_embed_dim + num_cont + 1 + sr_embed_dim

        self.seq_encoder = nn.GRU(
            input_size=seq_input_dim,
            hidden_size=cfg.seq_hidden_dim,
            num_layers=cfg.seq_num_layers,
            batch_first=True,
            bidirectional=cfg.seq_bidirectional,
            dropout=cfg.dropout if cfg.seq_num_layers > 1 else 0.0,
        )
        seq_out_dim = cfg.seq_hidden_dim * (2 if cfg.seq_bidirectional else 1)
        self.seq_out_dim = seq_out_dim

        # === 打者履歴エンコーダ (新規) ===
        # Inner GRU: 各打席の投球列 → 打席ベクトル
        # 入力: pitch_type_emb + cont + swing_attempt + swing_result_emb
        hist_pitch_input_dim = pt_embed_dim + num_cont + 1 + sr_embed_dim

        # 打席内投球シーケンスと共有する embedding を使用
        self.hist_inner_gru = nn.GRU(
            input_size=hist_pitch_input_dim,
            hidden_size=cfg.batter_hist_hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        # Inner GRU の出力に打席結果（bb_type emb + launch_speed + launch_angle）を結合
        bb_embed_dim = 4
        self.hist_bb_type_embed = nn.Embedding(cfg.num_bb_type + 1, bb_embed_dim)
        # 打席ベクトル = inner_hidden + bb_type_emb + launch_speed + launch_angle
        atbat_vec_dim = cfg.batter_hist_hidden_dim + bb_embed_dim + 2

        # Outer GRU: 打席ベクトル列 → 打者履歴ベクトル
        self.hist_outer_gru = nn.GRU(
            input_size=atbat_vec_dim,
            hidden_size=cfg.batter_hist_hidden_dim,
            num_layers=cfg.batter_hist_num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.batter_hist_num_layers > 1 else 0.0,
        )
        batter_hist_out_dim = cfg.batter_hist_hidden_dim
        self.batter_hist_out_dim = batter_hist_out_dim

        # === 共有バックボーン (ResBlocks) ===
        input_dim = embed_total_dim + num_cont + num_ord + seq_out_dim + batter_hist_out_dim

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
            layers.extend([nn.Linear(d, h), nn.GELU(), nn.Dropout(self.cfg.dropout)])
            d = h
        layers.append(nn.Linear(d, out_dim))
        return nn.Sequential(*layers)

    def _encode_pitch_sequence(
        self,
        seq_pitch_type: torch.Tensor,
        seq_cont: torch.Tensor,
        seq_swing_attempt: torch.Tensor,
        seq_swing_result: torch.Tensor,
        seq_mask: torch.Tensor,
    ) -> torch.Tensor:
        """打席内過去投球系列をエンコードする (B, T, ...) → (B, seq_out_dim)."""
        B, T = seq_pitch_type.shape
        device = seq_pitch_type.device

        num_pt = self.cfg.embedding_dims["pitch_type"][0]
        pt = seq_pitch_type.clamp(0, num_pt)
        pt_emb = self.seq_pitch_type_embed(pt)

        sr = seq_swing_result.clone()
        sr[sr < 0] = self.cfg.num_swing_result
        sr_emb = self.seq_swing_result_embed(sr)

        seq_feats = torch.cat(
            [pt_emb, seq_cont, seq_swing_attempt.unsqueeze(-1), sr_emb],
            dim=-1,
        )

        seq_lengths = seq_mask.sum(dim=1).long()
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
        h_n = torch.cat([h_n[-2], h_n[-1]], dim=-1) if self.cfg.seq_bidirectional else h_n[-1]
        seq_out[has_seq] = h_n
        return seq_out

    def _encode_batter_history(
        self,
        hist_pitch_type: torch.Tensor,  # (B, N, P)
        hist_cont: torch.Tensor,  # (B, N, P, num_cont)
        hist_swing_attempt: torch.Tensor,  # (B, N, P)
        hist_swing_result: torch.Tensor,  # (B, N, P)
        hist_bb_type: torch.Tensor,  # (B, N)
        hist_launch_speed: torch.Tensor,  # (B, N)
        hist_launch_angle: torch.Tensor,  # (B, N)
        hist_pitch_mask: torch.Tensor,  # (B, N, P)
        hist_atbat_mask: torch.Tensor,  # (B, N)
    ) -> torch.Tensor:
        """打者履歴を階層的にエンコードする → (B, batter_hist_out_dim)."""
        B, N, P = hist_pitch_type.shape
        device = hist_pitch_type.device

        # --- Inner GRU: 各打席の投球列を打席ベクトルに変換 ---
        # (B, N, P) → (B*N, P) にreshape してバッチ処理
        flat_pt = hist_pitch_type.reshape(B * N, P)
        flat_cont = hist_cont.reshape(B * N, P, -1)
        flat_sa = hist_swing_attempt.reshape(B * N, P)
        flat_sr = hist_swing_result.reshape(B * N, P)
        flat_pmask = hist_pitch_mask.reshape(B * N, P)

        # pitch_type embedding (シーケンスエンコーダと共有)
        num_pt = self.cfg.embedding_dims["pitch_type"][0]
        flat_pt = flat_pt.clamp(0, num_pt)
        pt_emb = self.seq_pitch_type_embed(flat_pt)  # (B*N, P, pt_embed_dim)

        # swing_result embedding (シーケンスエンコーダと共有)
        sr = flat_sr.clone()
        sr[sr < 0] = self.cfg.num_swing_result
        sr_emb = self.seq_swing_result_embed(sr)  # (B*N, P, sr_embed_dim)

        inner_feats = torch.cat(
            [pt_emb, flat_cont, flat_sa.unsqueeze(-1), sr_emb],
            dim=-1,
        )  # (B*N, P, hist_pitch_input_dim)

        # 有効な投球がある打席のみ処理
        inner_lengths = flat_pmask.sum(dim=1).long()  # (B*N,)
        has_pitches = inner_lengths > 0

        inner_out = torch.zeros(B * N, self.cfg.batter_hist_hidden_dim, device=device)

        if has_pitches.any():
            valid_feats = inner_feats[has_pitches]
            valid_lengths = inner_lengths[has_pitches]

            packed = nn.utils.rnn.pack_padded_sequence(
                valid_feats, valid_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, h_n = self.hist_inner_gru(packed)
            inner_out[has_pitches] = h_n[-1]

        inner_out = inner_out.reshape(B, N, self.cfg.batter_hist_hidden_dim)

        # --- 打席結果を結合 ---
        # bb_type embedding
        bb = hist_bb_type.clone()
        bb[bb < 0] = self.cfg.num_bb_type  # -1 (no bb) → padding idx
        bb_emb = self.hist_bb_type_embed(bb)  # (B, N, bb_embed_dim)

        atbat_vecs = torch.cat(
            [
                inner_out,
                bb_emb,
                hist_launch_speed.unsqueeze(-1),
                hist_launch_angle.unsqueeze(-1),
            ],
            dim=-1,
        )  # (B, N, atbat_vec_dim)

        # --- Outer GRU: 打席列 → 打者履歴ベクトル ---
        atbat_lengths = hist_atbat_mask.sum(dim=1).long()  # (B,)
        has_history = atbat_lengths > 0

        hist_out = torch.zeros(B, self.batter_hist_out_dim, device=device)

        if has_history.any():
            valid_vecs = atbat_vecs[has_history]
            valid_lengths = atbat_lengths[has_history]

            packed = nn.utils.rnn.pack_padded_sequence(
                valid_vecs, valid_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, h_n = self.hist_outer_gru(packed)
            hist_out[has_history] = h_n[-1]

        return hist_out

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
        hist_pitch_type: torch.Tensor | None = None,
        hist_cont: torch.Tensor | None = None,
        hist_swing_attempt: torch.Tensor | None = None,
        hist_swing_result: torch.Tensor | None = None,
        hist_bb_type: torch.Tensor | None = None,
        hist_launch_speed: torch.Tensor | None = None,
        hist_launch_angle: torch.Tensor | None = None,
        hist_pitch_mask: torch.Tensor | None = None,
        hist_atbat_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        B = cont.shape[0]
        device = cont.device

        # Embed categorical features
        embeds = []
        for feat_name in self.cfg.embedding_dims:
            x = cat_dict[feat_name]
            num_classes = self.cfg.embedding_dims[feat_name][0]
            x = torch.where((x < 0) | (x >= num_classes), num_classes, x)
            embeds.append(self.embeddings[feat_name](x))

        # 打席内投球シーケンスエンコード
        if seq_pitch_type is not None:
            seq_emb = self._encode_pitch_sequence(
                seq_pitch_type, seq_cont, seq_swing_attempt, seq_swing_result, seq_mask
            )
        else:
            seq_emb = torch.zeros(B, self.seq_out_dim, device=device)

        # 打者履歴エンコード
        if hist_pitch_type is not None:
            batter_hist_emb = self._encode_batter_history(
                hist_pitch_type,
                hist_cont,
                hist_swing_attempt,
                hist_swing_result,
                hist_bb_type,
                hist_launch_speed,
                hist_launch_angle,
                hist_pitch_mask,
                hist_atbat_mask,
            )
        else:
            batter_hist_emb = torch.zeros(B, self.batter_hist_out_dim, device=device)

        # 全入力を結合
        parts = embeds + [cont, ord_feat, seq_emb, batter_hist_emb]
        x = torch.cat(parts, dim=-1)

        # 共有バックボーン
        h = self.backbone(x)

        return {
            "swing_attempt": self.head_swing_attempt(h).squeeze(-1),
            "swing_result": self.head_swing_result(h),
            "bb_type": self.head_bb_type(h),
            "regression": self.head_regression(h),
        }
