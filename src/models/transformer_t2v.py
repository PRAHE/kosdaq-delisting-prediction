"""
S1 Phase 3 — Transformer + Time2Vec.

Time2Vec (Kazemi et al. 2019):
  τ_0(t) = ω_0 * t + φ_0           # 선형 성분
  τ_i(t) = sin(ω_i * t + φ_i)      # i ≥ 1, 사인 성분

K=3이라는 짧은 시퀀스에서는 self-attention의 장점이 크게 살지 않지만,
GRU 계열과의 ablation 차이가 명확해진다.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Time2Vec(nn.Module):
    """t (B, K) → (B, K, time_dim)."""
    def __init__(self, time_dim: int = 8):
        super().__init__()
        self.time_dim = time_dim
        # 초기화: 작은 omega로 시작
        self.omega = nn.Parameter(torch.randn(time_dim) * 0.3)
        self.phi = nn.Parameter(torch.zeros(time_dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B, K) float → broadcast to (B, K, time_dim)
        proj = t.unsqueeze(-1) * self.omega + self.phi  # (B, K, time_dim)
        # 첫 차원은 선형, 나머지는 sin
        linear = proj[..., :1]
        periodic = torch.sin(proj[..., 1:])
        return torch.cat([linear, periodic], dim=-1)


class TransformerT2VClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        time_dim: int = 8,
        model_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 1,
        ffn_mult: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.time2vec = Time2Vec(time_dim)
        self.input_proj = nn.Linear(input_dim + time_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ffn_mult * model_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.head = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 2, 1),
        )

    def forward(self, X: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        """X: (B, K, D), M: (B, K) (1=observed, 0=padded)."""
        B, K, _ = X.shape
        # time index 0..K-1
        t = torch.arange(K, dtype=X.dtype, device=X.device).expand(B, K)
        t2v = self.time2vec(t)  # (B, K, time_dim)
        h = self.input_proj(torch.cat([X, t2v], dim=-1))  # (B, K, model_dim)

        # padding mask: True = 무시할 위치
        key_padding_mask = (M == 0)  # (B, K)
        # 만약 모든 위치가 mask이면 attention NaN — 그 경우 첫 위치를 살림
        all_masked = key_padding_mask.all(dim=1)
        if all_masked.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_masked, 0] = False

        h = self.encoder(h, src_key_padding_mask=key_padding_mask)

        # Pooling: 관측된 위치의 평균
        m = M.float().unsqueeze(-1)  # (B, K, 1)
        h_pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        return self.head(h_pooled).squeeze(-1)
