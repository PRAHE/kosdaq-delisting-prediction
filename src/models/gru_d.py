"""
GRU-D (Che et al. 2018, Scientific Reports).

Input:
  X (B, K, D)       — raw values (결측 자리 0)
  M (B, K, D)       — 관측 여부 (1/0)
  X_last (B, K, D)  — 각 (t,d)에서의 가장 최근 관측값 (없으면 x̄)
  Delta (B, K, D)   — 결측 timestep 거리 (관측 = 0)

Decay:
  γ^x_t = exp(-ReLU(W_γx · δ_t + b_γx))   W_γx, b_γx ∈ R^D, element-wise
  γ^h_t = exp(-ReLU(W_γh · δ_t + b_γh))   W_γh ∈ R^{D×H}, b_γh ∈ R^H
  → γ^h_t ∈ R^H

Imputation:
  x̂_t = M_t * X_t + (1 - M_t) * (γ^x_t * X_last_t + (1 - γ^x_t) * x̄)

Hidden decay:
  h̃_{t-1} = γ^h_t ⊙ h_{t-1}

GRU step: GRUCell([x̂_t; M_t], h̃_{t-1})
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUDClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.2,
        x_mean: np.ndarray | None = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # x_mean buffer (D,)
        if x_mean is None:
            x_mean = np.zeros(input_dim, dtype=np.float32)
        x_mean_t = torch.from_numpy(np.asarray(x_mean, dtype=np.float32))
        # NaN safe
        x_mean_t = torch.nan_to_num(x_mean_t, nan=0.0)
        self.register_buffer("x_mean", x_mean_t)

        # Input decay: D 차원
        self.W_dec_x = nn.Parameter(torch.zeros(input_dim))
        self.b_dec_x = nn.Parameter(torch.zeros(input_dim))

        # Hidden decay: D → H
        self.W_dec_h = nn.Linear(input_dim, hidden_dim, bias=True)

        # GRUCell: 입력 = [x̂_t; m_t] = 2D
        self.cell = nn.GRUCell(2 * input_dim, hidden_dim)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        X: torch.Tensor,       # (B, K, D)
        M: torch.Tensor,       # (B, K, D)
        X_last: torch.Tensor,  # (B, K, D)
        Delta: torch.Tensor,   # (B, K, D)
    ) -> torch.Tensor:
        B, K, D = X.shape
        h = X.new_zeros(B, self.hidden_dim)
        x_mean = self.x_mean.view(1, D).expand(B, D)

        for k in range(K):
            x_t = X[:, k, :]
            m_t = M[:, k, :]
            xl_t = X_last[:, k, :]
            d_t = Delta[:, k, :]

            # Input decay (feature-wise)
            # γ^x_t = exp(-ReLU(W_γx * δ_t + b_γx))
            g_x = torch.exp(-F.relu(d_t * self.W_dec_x + self.b_dec_x))  # (B, D)

            # Imputation
            x_imp = (1 - m_t) * (g_x * xl_t + (1 - g_x) * x_mean) + m_t * x_t  # (B, D)

            # Hidden decay
            # γ^h_t = exp(-ReLU(W_γh · δ_t + b_γh))
            g_h = torch.exp(-F.relu(self.W_dec_h(d_t)))  # (B, H)
            h = g_h * h

            # GRU step
            inp = torch.cat([x_imp, m_t], dim=-1)  # (B, 2D)
            h = self.cell(inp, h)

        return self.head(h).squeeze(-1)


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> torch.Tensor:
    p = torch.sigmoid(logits)
    pt = torch.where(targets == 1, p, 1 - p)
    a = torch.where(targets == 1, torch.tensor(alpha, device=logits.device),
                    torch.tensor(1 - alpha, device=logits.device))
    return (-a * (1 - pt).clamp(min=1e-8).pow(gamma) * pt.clamp(min=1e-8).log()).mean()
