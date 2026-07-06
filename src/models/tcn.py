"""
S1 시계열 — TCN (Temporal Convolutional Network, Bai et al. 2018) classifier.

Dilated causal Conv1d 스택 + mask-aware mean pooling + 분류 head. K=3 짧은
시퀀스라 dilation [1,2], kernel 2로 receptive field를 덮는다. focal loss / Adam /
ReduceLROnPlateau은 gru_baseline에서 재사용 — 시계열 모델군 동일 조건 비교.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.gru import TrainResult, focal_loss, to_tensors


class _Chomp1d(nn.Module):
    """causal conv의 우측 padding 제거."""
    def __init__(self, chomp: int):
        super().__init__()
        self.chomp = chomp

    def forward(self, x):
        return x[:, :, :-self.chomp].contiguous() if self.chomp > 0 else x


class _TemporalBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel, dilation, dropout):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, kernel, padding=pad, dilation=dilation),
            _Chomp1d(pad), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(c_out, c_out, kernel, padding=pad, dilation=dilation),
            _Chomp1d(pad), nn.ReLU(), nn.Dropout(dropout),
        )
        self.down = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.down is None else self.down(x)
        return self.relu(out + res)


class TCNClassifier(nn.Module):
    def __init__(self, input_dim: int, channels: int = 64, levels: int = 2,
                 kernel: int = 2, dropout: float = 0.2):
        super().__init__()
        blocks = []
        c_in = input_dim
        for i in range(levels):
            blocks.append(_TemporalBlock(c_in, channels, kernel, dilation=2 ** i, dropout=dropout))
            c_in = channels
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Linear(channels, channels // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(channels // 2, 1),
        )

    def forward(self, X: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        # X: (B, K, F) → (B, F, K)
        h = self.tcn(X.transpose(1, 2))          # (B, C, K)
        h = h.transpose(1, 2)                     # (B, K, C)
        m = M.unsqueeze(-1).float()               # (B, K, 1)
        pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)  # mask-aware mean
        return self.head(pooled).squeeze(-1)


def _eval(model, loader, device):
    model.eval(); out = []
    with torch.no_grad():
        for X, M, _ in loader:
            out.append(torch.sigmoid(model(X.to(device), M.to(device))).cpu().numpy())
    return np.concatenate(out)


def train_tcn(
    train_X, train_M, train_y, valid_X, valid_M, valid_y, test_X, test_M, test_y,
    channels: int = 64, levels: int = 2, kernel: int = 2, dropout: float = 0.2,
    batch_size: int = 256, lr: float = 1e-3, epochs: int = 30, patience: int = 8,
    focal_gamma: float = 2.0, focal_alpha: float = 0.25, seed: int = 42,
    verbose: bool = False,
) -> TrainResult:
    from sklearn.metrics import average_precision_score
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cpu")
    model = TCNClassifier(train_X.shape[-1], channels, levels, kernel, dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)

    tr = DataLoader(TensorDataset(*to_tensors(train_X, train_M, train_y)), batch_size=batch_size, shuffle=True)
    vl = DataLoader(TensorDataset(*to_tensors(valid_X, valid_M, valid_y)), batch_size=batch_size, shuffle=False)
    te = DataLoader(TensorDataset(*to_tensors(test_X, test_M, test_y)), batch_size=batch_size, shuffle=False)

    best_pr, best_ep, best_state, no_imp = -1.0, -1, None, 0
    train_losses, valid_prs = [], []
    for ep in range(1, epochs + 1):
        model.train(); losses = []
        for X, M, y in tr:
            X, M, y = X.to(device), M.to(device), y.to(device)
            opt.zero_grad()
            loss = focal_loss(model(X, M), y, gamma=focal_gamma, alpha=focal_alpha)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); losses.append(float(loss.item()))
        train_losses.append(float(np.mean(losses)))
        vproba = _eval(model, vl, device)
        vpr = float(average_precision_score(valid_y, vproba))
        valid_prs.append(vpr); sched.step(vpr)
        if vpr > best_pr:
            best_pr, best_ep = vpr, ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainResult(best_valid_pr_auc=best_pr, best_epoch=best_ep,
                       train_losses=train_losses, valid_pr_aucs=valid_prs,
                       test_proba=_eval(model, te, device),
                       valid_proba=_eval(model, vl, device))
