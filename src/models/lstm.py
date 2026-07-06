"""
S1 시계열 — LSTM baseline classifier (GRU의 LSTM 변형).

GRUClassifier와 동일 구조에서 nn.GRU → nn.LSTM만 교체. 마스크로 마지막 관측
timestep의 hidden state를 분류 입력으로. focal loss / Adam / ReduceLROnPlateau은
gru_baseline에서 재사용 — 시계열 모델군 비교의 동일 조건 유지.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.gru import TrainResult, focal_loss, to_tensors, _eval_model


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 1,
                 dropout: float = 0.2, use_mask: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_mask = use_mask
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, X: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        H, _ = self.lstm(X)  # (B, K, hidden)
        if self.use_mask:
            lens = M.sum(dim=1).clamp(min=1) - 1
            idx = lens.long().unsqueeze(1).unsqueeze(2).expand(-1, 1, self.hidden_dim)
            last = H.gather(1, idx).squeeze(1)
        else:
            last = H[:, -1, :]
        return self.head(last).squeeze(-1)


def train_lstm(
    train_X, train_M, train_y, valid_X, valid_M, valid_y, test_X, test_M, test_y,
    hidden_dim: int = 64, num_layers: int = 1, dropout: float = 0.2,
    batch_size: int = 256, lr: float = 1e-3, epochs: int = 30, patience: int = 8,
    focal_gamma: float = 2.0, focal_alpha: float = 0.25, seed: int = 42,
    verbose: bool = False,
) -> TrainResult:
    from sklearn.metrics import average_precision_score
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cpu")
    model = LSTMClassifier(train_X.shape[-1], hidden_dim, num_layers, dropout).to(device)
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
        vproba = _eval_model(model, vl, device)
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
                       test_proba=_eval_model(model, te, device),
                       valid_proba=_eval_model(model, vl, device))
