"""
S1 Phase 1 — GRU baseline classifier.

(N, K, F) 시퀀스 입력 → GRU → 마지막 hidden state → linear → sigmoid.
mask는 packed sequence가 아닌 length로 처리 (간단).

Loss: focal loss (Lin et al. 2017, γ=2) — 극심한 불균형에 BCE보다 안정.
Optimizer: Adam.
Scheduler: ReduceLROnPlateau on valid PR-AUC.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class GRUClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
        use_mask: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_mask = use_mask
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, X: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        """X: (B, K, F), M: (B, K) int.

        Mask 사용 시: 마지막 관측 timestep의 hidden state를 분류 입력으로.
        """
        H, _ = self.gru(X)  # (B, K, hidden)
        if self.use_mask:
            # 마지막 1 위치 index — M.sum(-1)-1 = 마지막 관측의 K index
            lens = M.sum(dim=1).clamp(min=1) - 1  # (B,)
            last_idx = lens.long().unsqueeze(1).unsqueeze(2).expand(-1, 1, self.hidden_dim)
            last = H.gather(1, last_idx).squeeze(1)  # (B, hidden)
        else:
            last = H[:, -1, :]
        return self.head(last).squeeze(-1)  # (B,)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> torch.Tensor:
    """Lin et al. 2017 binary focal loss. targets 0/1 float."""
    p = torch.sigmoid(logits)
    pt = torch.where(targets == 1, p, 1 - p)
    a = torch.where(targets == 1, torch.tensor(alpha, device=logits.device),
                    torch.tensor(1 - alpha, device=logits.device))
    return (-a * (1 - pt).clamp(min=1e-8).pow(gamma) * pt.clamp(min=1e-8).log()).mean()


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------


@dataclass
class TrainResult:
    best_valid_pr_auc: float
    best_epoch: int
    train_losses: list
    valid_pr_aucs: list
    test_proba: np.ndarray
    valid_proba: np.ndarray


def to_tensors(ds_X: np.ndarray, ds_M: np.ndarray, ds_y: np.ndarray):
    X = torch.from_numpy(ds_X)
    M = torch.from_numpy(ds_M.astype(np.float32))
    y = torch.from_numpy(ds_y.astype(np.float32))
    return X, M, y


def _eval_model(model, loader, device):
    model.eval()
    probas = []
    with torch.no_grad():
        for X, M, _ in loader:
            X = X.to(device); M = M.to(device)
            logits = model(X, M)
            probas.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probas)


def train_gru(
    train_X, train_M, train_y,
    valid_X, valid_M, valid_y,
    test_X, test_M, test_y,
    hidden_dim: int = 64,
    num_layers: int = 1,
    dropout: float = 0.2,
    batch_size: int = 256,
    lr: float = 1e-3,
    epochs: int = 30,
    patience: int = 8,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.25,
    seed: int = 42,
    verbose: bool = True,
) -> TrainResult:
    from sklearn.metrics import average_precision_score

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")

    F_ = train_X.shape[-1]
    model = GRUClassifier(F_, hidden_dim, num_layers, dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=3,
    )

    tr_ds = TensorDataset(*to_tensors(train_X, train_M, train_y))
    vl_ds = TensorDataset(*to_tensors(valid_X, valid_M, valid_y))
    te_ds = TensorDataset(*to_tensors(test_X, test_M, test_y))
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    vl_loader = DataLoader(vl_ds, batch_size=batch_size, shuffle=False)
    te_loader = DataLoader(te_ds, batch_size=batch_size, shuffle=False)

    best_pr = -1.0
    best_ep = -1
    best_state = None
    train_losses = []
    valid_prs = []
    no_improve = 0

    for ep in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        for X, M, y in tr_loader:
            X = X.to(device); M = M.to(device); y = y.to(device)
            opt.zero_grad()
            logits = model(X, M)
            loss = focal_loss(logits, y, gamma=focal_gamma, alpha=focal_alpha)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            epoch_losses.append(float(loss.item()))
        avg_loss = float(np.mean(epoch_losses))
        train_losses.append(avg_loss)

        # valid PR-AUC
        valid_proba = _eval_model(model, vl_loader, device)
        valid_pr = float(average_precision_score(valid_y, valid_proba))
        valid_prs.append(valid_pr)
        scheduler.step(valid_pr)

        if verbose:
            print(f"  ep {ep:>2}  loss={avg_loss:.4f}  valid_pr={valid_pr:.4f}  best={best_pr:.4f}")

        if valid_pr > best_pr:
            best_pr = valid_pr
            best_ep = ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"  early stop @ epoch {ep}")
                break

    # 최적 state로 test 평가
    if best_state is not None:
        model.load_state_dict(best_state)
    test_proba = _eval_model(model, te_loader, device)
    valid_proba = _eval_model(model, vl_loader, device)

    return TrainResult(
        best_valid_pr_auc=best_pr,
        best_epoch=best_ep,
        train_losses=train_losses,
        valid_pr_aucs=valid_prs,
        test_proba=test_proba,
        valid_proba=valid_proba,
    )
