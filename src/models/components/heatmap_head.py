"""ヒートマップベースの回帰ヘッド."""

import torch
import torch.nn as nn

from config import ModelConfig
from models.components.heatmap_utils import make_heatmap_key


class Heatmap2DSubHead(nn.Module):
    """launch_angle × spray_angle の 2D ヒートマップ + オフセットサブヘッド.

    backbone 出力（1D ベクトル）から deconvolution で 2D ヒートマップを生成する。
    ヒートマップブランチとオフセットブランチを持つ。
    """

    def __init__(
        self,
        in_dim: int,
        grid_h: int = 64,
        grid_w: int = 64,
        intermediate_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w

        # 初期空間サイズ: 4×4 から deconv で拡大
        self.init_h = 4
        self.init_w = 4

        # 必要な deconv 段数を計算 (4→8→16→32→64 = 4段)
        self.num_deconv = 0
        h, w = self.init_h, self.init_w
        while h < grid_h or w < grid_w:
            h *= 2
            w *= 2
            self.num_deconv += 1

        # Linear: backbone_out → intermediate_dim * init_h * init_w
        self.fc = nn.Sequential(
            nn.Linear(in_dim, intermediate_dim * self.init_h * self.init_w),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Shared deconv trunk
        channels = [intermediate_dim]
        for i in range(self.num_deconv):
            channels.append(max(16, intermediate_dim // (2 ** (i + 1))))

        deconv_layers: list[nn.Module] = []
        for i in range(self.num_deconv):
            deconv_layers.extend([
                nn.ConvTranspose2d(channels[i], channels[i + 1], kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(channels[i + 1]),
                nn.ReLU(),
            ])
        self.deconv = nn.Sequential(*deconv_layers)

        last_ch = channels[-1]

        # Heatmap branch: Conv2d → sigmoid
        self.heatmap_conv = nn.Conv2d(last_ch, 1, kernel_size=3, padding=1)

        # Offset branch: Conv2d → (dy, dx)
        self.offset_conv = nn.Conv2d(last_ch, 2, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: (B, in_dim) backbone 出力

        Returns:
            heatmap: (B, 1, H, W) — sigmoid 適用済み
            offset: (B, 2, H, W)
        """
        B = x.size(0)
        h = self.fc(x)  # (B, C * init_h * init_w)
        h = h.view(B, -1, self.init_h, self.init_w)  # (B, C, 4, 4)
        h = self.deconv(h)  # (B, last_ch, H', W')

        # グリッドサイズに合わせてクロップ（deconv の出力が大きい場合）
        h = h[:, :, :self.grid_h, :self.grid_w]

        heatmap = torch.sigmoid(self.heatmap_conv(h))  # (B, 1, H, W)
        offset = self.offset_conv(h)                    # (B, 2, H, W)

        return heatmap, offset


class Heatmap1DSubHead(nn.Module):
    """1D ヒートマップ + オフセットサブヘッド.

    launch_speed や hit_distance_sc の予測に使用する。
    """

    def __init__(
        self,
        in_dim: int,
        num_bins: int = 64,
        intermediate_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_bins = num_bins

        # 初期サイズ: 4 から deconv で拡大
        self.init_len = 4

        self.num_deconv = 0
        length = self.init_len
        while length < num_bins:
            length *= 2
            self.num_deconv += 1

        # Linear: backbone_out → intermediate_dim * init_len
        self.fc = nn.Sequential(
            nn.Linear(in_dim, intermediate_dim * self.init_len),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Shared deconv trunk
        channels = [intermediate_dim]
        for i in range(self.num_deconv):
            channels.append(max(16, intermediate_dim // (2 ** (i + 1))))

        deconv_layers: list[nn.Module] = []
        for i in range(self.num_deconv):
            deconv_layers.extend([
                nn.ConvTranspose1d(channels[i], channels[i + 1], kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(channels[i + 1]),
                nn.ReLU(),
            ])
        self.deconv = nn.Sequential(*deconv_layers)

        last_ch = channels[-1]

        # Heatmap branch
        self.heatmap_conv = nn.Conv1d(last_ch, 1, kernel_size=3, padding=1)

        # Offset branch
        self.offset_conv = nn.Conv1d(last_ch, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: (B, in_dim) backbone 出力

        Returns:
            heatmap: (B, 1, L) — sigmoid 適用済み
            offset: (B, 1, L)
        """
        B = x.size(0)
        h = self.fc(x)  # (B, C * init_len)
        h = h.view(B, -1, self.init_len)  # (B, C, 4)
        h = self.deconv(h)  # (B, last_ch, L')

        # ビン数に合わせてクロップ
        h = h[:, :, :self.num_bins]

        heatmap = torch.sigmoid(self.heatmap_conv(h))  # (B, 1, L)
        offset = self.offset_conv(h)                    # (B, 1, L)

        return heatmap, offset


class HeatmapHead(nn.Module):
    """ヒートマップ回帰ヘッド.

    heatmap_heads が設定されている場合（設定モード）:
        YAML で指定された任意の 1D/2D サブヘッド構成を動的に構築する。

    heatmap_heads が未設定の場合（レガシーモード）:
        従来のハードコード構成を使用:
        - head_2d: launch_angle × spray_angle の 2D ヒートマップ
        - head_launch_speed: launch_speed の 1D ヒートマップ
        - head_hit_distance: hit_distance_sc の 1D ヒートマップ
    """

    def __init__(self, in_dim: int, cfg: ModelConfig):
        super().__init__()
        head_configs = cfg.get_heatmap_head_configs()
        self._legacy_mode = head_configs is None

        if self._legacy_mode:
            # レガシーモード: 従来のハードコードサブヘッド
            self.head_2d = Heatmap2DSubHead(
                in_dim=in_dim,
                grid_h=cfg.heatmap_grid_h,
                grid_w=cfg.heatmap_grid_w,
                intermediate_dim=cfg.heatmap_intermediate_dim,
                dropout=cfg.dropout,
            )
            self.head_launch_speed = Heatmap1DSubHead(
                in_dim=in_dim,
                num_bins=cfg.heatmap_num_bins,
                intermediate_dim=cfg.heatmap_intermediate_dim,
                dropout=cfg.dropout,
            )
            self.head_hit_distance = Heatmap1DSubHead(
                in_dim=in_dim,
                num_bins=cfg.heatmap_num_bins,
                intermediate_dim=cfg.heatmap_intermediate_dim,
                dropout=cfg.dropout,
            )
        else:
            # 設定モード: YAML 設定に基づく動的サブヘッド構築
            self.sub_heads = nn.ModuleDict()
            self._head_meta: list[tuple[str, str]] = []  # (key, type)
            for hc in head_configs:
                key = make_heatmap_key(hc.type, hc.targets)
                if hc.type == "2d":
                    grid_h = hc.grid_h if hc.grid_h is not None else cfg.heatmap_grid_h
                    grid_w = hc.grid_w if hc.grid_w is not None else cfg.heatmap_grid_w
                    self.sub_heads[key] = Heatmap2DSubHead(
                        in_dim=in_dim,
                        grid_h=grid_h,
                        grid_w=grid_w,
                        intermediate_dim=cfg.heatmap_intermediate_dim,
                        dropout=cfg.dropout,
                    )
                else:
                    num_bins = hc.num_bins if hc.num_bins is not None else cfg.heatmap_num_bins
                    self.sub_heads[key] = Heatmap1DSubHead(
                        in_dim=in_dim,
                        num_bins=num_bins,
                        intermediate_dim=cfg.heatmap_intermediate_dim,
                        dropout=cfg.dropout,
                    )
                self._head_meta.append((key, hc.type))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: (B, in_dim) backbone 出力

        Returns:
            レガシーモード:
                heatmap_2d, offset_2d, heatmap_launch_speed, offset_launch_speed,
                heatmap_hit_distance, offset_hit_distance
            設定モード:
                heatmap_{key}, offset_{key} （key は make_heatmap_key で生成）
        """
        if self._legacy_mode:
            hm_2d, off_2d = self.head_2d(x)
            hm_ls, off_ls = self.head_launch_speed(x)
            hm_hd, off_hd = self.head_hit_distance(x)
            return {
                "heatmap_2d": hm_2d,
                "offset_2d": off_2d,
                "heatmap_launch_speed": hm_ls,
                "offset_launch_speed": off_ls,
                "heatmap_hit_distance": hm_hd,
                "offset_hit_distance": off_hd,
            }

        out: dict[str, torch.Tensor] = {}
        for key, htype in self._head_meta:
            hm, off = self.sub_heads[key](x)
            out[f"heatmap_{key}"] = hm
            out[f"offset_{key}"] = off
        return out
