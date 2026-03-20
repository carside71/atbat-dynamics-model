"""Head コンポーネント."""

import torch
import torch.nn as nn

ACTIVATION_MAP = {"relu": nn.ReLU, "gelu": nn.GELU}


def build_mlp_head(
    in_dim: int, hidden_dims: list[int], out_dim: int, dropout: float, activation: str = "gelu"
) -> nn.Sequential:
    """MLP ヘッドを構築する."""
    act_cls = ACTIVATION_MAP[activation]
    layers: list[nn.Module] = []
    d = in_dim
    for h in hidden_dims:
        layers.extend([nn.Linear(d, h), act_cls(), nn.Dropout(dropout)])
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class MDNHead(nn.Module):
    """Mixture Density Network ヘッド.

    K 個のガウス成分を用いて D 次元の出力を分布として予測する。
    出力: dict with pi (B, K), mu (B, K, D), sigma (B, K, D)
    """

    def __init__(self, in_dim: int, hidden_dims: list[int], out_dim: int, num_components: int, dropout: float):
        super().__init__()
        self.out_dim = out_dim
        self.num_components = num_components

        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)])
            d = h
        self.shared = nn.Sequential(*layers)

        self.fc_pi = nn.Linear(d, num_components)
        self.fc_mu = nn.Linear(d, num_components * out_dim)
        self.fc_sigma = nn.Linear(d, num_components * out_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.shared(x)
        B = x.size(0)
        K, D = self.num_components, self.out_dim

        pi = torch.softmax(self.fc_pi(h), dim=-1)
        mu = self.fc_mu(h).view(B, K, D)
        sigma = nn.functional.softplus(self.fc_sigma(h)).view(B, K, D) + 1e-6

        return {"pi": pi, "mu": mu, "sigma": sigma}
