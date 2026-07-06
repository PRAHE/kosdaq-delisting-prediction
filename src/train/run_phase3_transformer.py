"""
S1 Phase 3 — Transformer + Time2Vec.

Phase 1 GRU와 동일 데이터(이미 imputed)를 사용. K=3 짧은 시퀀스에서
self-attention이 GRU 대비 우위를 가지는지 검증.

출력:
  results/research_s1_phase3/
    P3_test_multi_seed.csv
    P3_summary.json
    P3_pr_auc_comparison.png       # 4-way: Transformer, GRU, GRU-D, RF
    P3_train_curves.png
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from src.research.s0_diagnostic.baseline import PROJECT_ROOT, prepare_baseline, train_baseline_rf
from src.evaluate.bayesian_bootstrap import bayesian_bootstrap, to_dict

from src.models.gru import focal_loss
from src.models.sequences import prepare_phase1_datasets, summarize
from src.models.transformer_t2v import TransformerT2VClassifier

OUT_DIR = PROJECT_ROOT / "results" / "research_s1_phase3"


def _top_k_precision(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    if k <= 0:
        return float("nan")
    idx = np.argsort(scores)[::-1][:k]
    return float(np.mean(y_true[idx]))


@dataclass
class TrainResult:
    best_valid_pr_auc: float
    best_epoch: int
    train_losses: list
    valid_pr_aucs: list
    test_proba: np.ndarray


def _to_tensors(X, M, y):
    return torch.from_numpy(X), torch.from_numpy(M.astype(np.float32)), torch.from_numpy(y.astype(np.float32))


def _eval(model, loader, device):
    model.eval()
    out = []
    with torch.no_grad():
        for X, M, _ in loader:
            X = X.to(device); M = M.to(device)
            logits = model(X, M)
            out.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(out)


def train_transformer(
    ds_train, ds_valid, ds_test,
    model_dim: int = 64,
    num_heads: int = 4,
    num_layers: int = 1,
    dropout: float = 0.2,
    time_dim: int = 8,
    batch_size: int = 256,
    lr: float = 5e-4,
    epochs: int = 30,
    patience: int = 8,
    seed: int = 42,
    verbose: bool = False,
) -> TrainResult:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")

    D = ds_train.X.shape[-1]
    model = TransformerT2VClassifier(
        input_dim=D, time_dim=time_dim, model_dim=model_dim,
        num_heads=num_heads, num_layers=num_layers, dropout=dropout,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)

    tr_ds = TensorDataset(*_to_tensors(ds_train.X, ds_train.M, (ds_train.y > 0).astype(np.float32)))
    vl_ds = TensorDataset(*_to_tensors(ds_valid.X, ds_valid.M, (ds_valid.y > 0).astype(np.float32)))
    te_ds = TensorDataset(*_to_tensors(ds_test.X,  ds_test.M,  (ds_test.y  > 0).astype(np.float32)))
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    vl_loader = DataLoader(vl_ds, batch_size=batch_size, shuffle=False)
    te_loader = DataLoader(te_ds, batch_size=batch_size, shuffle=False)

    best_pr = -1.0; best_ep = -1; best_state = None
    train_losses = []; valid_prs = []; no_imp = 0
    y_valid = (ds_valid.y > 0).astype(int)

    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = []
        for X, M, y in tr_loader:
            X = X.to(device); M = M.to(device); y = y.to(device)
            opt.zero_grad()
            logits = model(X, M)
            loss = focal_loss(logits, y, gamma=2.0, alpha=0.25)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ep_loss.append(float(loss.item()))
        avg = float(np.mean(ep_loss))
        train_losses.append(avg)

        vp = _eval(model, vl_loader, device)
        vpr = float(average_precision_score(y_valid, vp))
        valid_prs.append(vpr)
        scheduler.step(vpr)

        if verbose:
            print(f"  ep {ep:>2}  loss={avg:.4f}  valid_pr={vpr:.4f}  best={best_pr:.4f}")

        if vpr > best_pr:
            best_pr = vpr; best_ep = ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                if verbose:
                    print(f"  early stop @ epoch {ep}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    tp = _eval(model, te_loader, device)
    return TrainResult(best_pr, best_ep, train_losses, valid_prs, tp)


def run_phase3(
    labeling: str = "L3_rolling_H24",
    K: int = 3,
    seeds=(42, 7, 13, 21, 100),
    model_dim: int = 64,
    num_heads: int = 4,
    num_layers: int = 1,
    dropout: float = 0.2,
    epochs: int = 30,
) -> dict:
    warnings.filterwarnings("ignore")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[S1 Phase 3 Transformer+T2V] labeling={labeling}  K={K}  seeds={list(seeds)}")
    data = prepare_phase1_datasets(labeling=labeling, K=K)
    print("  " + summarize(data["train"], "train"))
    print("  " + summarize(data["valid"], "valid"))
    print("  " + summarize(data["test"],  "test "))

    tr, vl, te = data["train"], data["valid"], data["test"]

    seed_rows = []; test_probas = []; train_curves = []; valid_curves = []
    for seed in seeds:
        print(f"\n--- seed={seed} ---")
        r = train_transformer(tr, vl, te,
                                model_dim=model_dim, num_heads=num_heads,
                                num_layers=num_layers, dropout=dropout,
                                epochs=epochs, seed=seed)
        y_test = (te.y > 0).astype(int)
        tpr = float(average_precision_score(y_test, r.test_proba))
        troc = float(roc_auc_score(y_test, r.test_proba))
        p20 = _top_k_precision(y_test, r.test_proba, 20)
        p50 = _top_k_precision(y_test, r.test_proba, 50)
        seed_rows.append({
            "seed": seed, "best_epoch": r.best_epoch,
            "valid_pr_auc": r.best_valid_pr_auc,
            "test_pr_auc": tpr, "test_roc_auc": troc,
            "test_p20": p20, "test_p50": p50,
        })
        test_probas.append(r.test_proba)
        train_curves.append(r.train_losses)
        valid_curves.append(r.valid_pr_aucs)
        print(f"  best_ep={r.best_epoch}  valid_pr={r.best_valid_pr_auc:.4f}  "
              f"test_pr={tpr:.4f}  test_roc={troc:.4f}  P@20={p20:.3f}")

    avg_proba = np.mean(np.stack(test_probas), axis=0)
    y_test = (te.y > 0).astype(int)
    bb = bayesian_bootstrap(y_test, avg_proba, metric="pr_auc", n_boot=2000, seed=42)

    # RF baseline
    print("\n[ref] RF baseline (S0 canonical)")
    rf_data = prepare_baseline()
    rf_test_probas = []; rf_rows = []
    for seed in seeds:
        rf = train_baseline_rf(rf_data.X_train, rf_data.y_train, seed=seed)
        p = rf.predict_proba(rf_data.X_test)[:, 1]
        rf_test_probas.append(p)
        rf_rows.append({
            "seed": seed,
            "test_pr_auc": float(average_precision_score(rf_data.y_test, p)),
            "test_roc_auc": float(roc_auc_score(rf_data.y_test, p)),
            "test_p20": _top_k_precision(rf_data.y_test, p, 20),
            "test_p50": _top_k_precision(rf_data.y_test, p, 50),
        })
    rf_avg = np.mean(np.stack(rf_test_probas), axis=0)
    bb_rf = bayesian_bootstrap(rf_data.y_test, rf_avg, metric="pr_auc", n_boot=2000, seed=42)

    # Phase 1, 2 결과 로드
    p1 = {}
    p2 = {}
    p1_path = PROJECT_ROOT / "results" / "research_s1_phase1" / "P1_summary.json"
    p2_path = PROJECT_ROOT / "results" / "research_s1_phase2" / "P2_summary.json"
    if p1_path.exists():
        with open(p1_path, encoding="utf-8") as f:
            p1 = json.load(f).get("GRU", {})
    if p2_path.exists():
        with open(p2_path, encoding="utf-8") as f:
            p2 = json.load(f).get("GRU_D", {})

    df_seeds = pd.DataFrame(seed_rows)
    df_rf = pd.DataFrame(rf_rows)
    df_seeds.to_csv(OUT_DIR / "P3_test_multi_seed.csv", index=False)
    df_rf.to_csv(OUT_DIR / "P3_rf_baseline_multi_seed.csv", index=False)

    summary = {
        "experiment": "S1_Phase3_Transformer_T2V",
        "labeling": labeling, "K": K, "seeds": list(seeds),
        "Transformer_T2V": {
            "test_pr_auc_mean": float(df_seeds["test_pr_auc"].mean()),
            "test_pr_auc_std":  float(df_seeds["test_pr_auc"].std()),
            "test_pr_auc_min":  float(df_seeds["test_pr_auc"].min()),
            "test_pr_auc_max":  float(df_seeds["test_pr_auc"].max()),
            "test_roc_auc_mean": float(df_seeds["test_roc_auc"].mean()),
            "test_p20_mean":     float(df_seeds["test_p20"].mean()),
            "test_p50_mean":     float(df_seeds["test_p50"].mean()),
            "bootstrap_ci_avg_proba": to_dict(bb),
            "model_config": {
                "model_dim": model_dim, "num_heads": num_heads, "num_layers": num_layers,
                "dropout": dropout, "lr": 5e-4, "batch_size": 256,
            },
        },
        "RF_baseline": {
            "test_pr_auc_mean": float(df_rf["test_pr_auc"].mean()),
            "test_pr_auc_std":  float(df_rf["test_pr_auc"].std()),
            "test_roc_auc_mean": float(df_rf["test_roc_auc"].mean()),
            "test_p20_mean":     float(df_rf["test_p20"].mean()),
            "test_p50_mean":     float(df_rf["test_p50"].mean()),
            "bootstrap_ci_avg_proba": to_dict(bb_rf),
        },
        "Phase1_GRU_ref":   p1,
        "Phase2_GRUD_ref":  p2,
    }
    with open(OUT_DIR / "P3_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 4-way 비교 그림
    items = [
        ("Transformer+T2V (P3)", summary["Transformer_T2V"]["test_pr_auc_mean"], bb,                   "#9B59B6"),
    ]
    if p1:
        items.append(("GRU (P1)", p1["test_pr_auc_mean"], p1["bootstrap_ci_avg_proba"], "#55A868"))
    if p2:
        items.append(("GRU-D (P2)", p2["test_pr_auc_mean"], p2["bootstrap_ci_avg_proba"], "#C44E52"))
    items.append(("RF baseline", summary["RF_baseline"]["test_pr_auc_mean"], bb_rf, "#4C72B0"))

    labels = [t[0] for t in items]
    means  = [t[1] for t in items]
    boots  = [t[2] for t in items]
    colors = [t[3] for t in items]

    err_lo = []; err_hi = []; lows = []; highs = []
    for m, b in zip(means, boots):
        lo = b["ci95_lo"] if isinstance(b, dict) else b.ci95_lo
        hi = b["ci95_hi"] if isinstance(b, dict) else b.ci95_hi
        err_lo.append(max(m - lo, 0)); err_hi.append(max(hi - m, 0))
        lows.append(lo); highs.append(hi)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=[err_lo, err_hi], color=colors, capsize=6, edgecolor="k")
    for i, (m, lo, hi) in enumerate(zip(means, lows, highs)):
        ax.text(i, hi + 0.005, f"{m:.4f}\nCI[{lo:.3f},{hi:.3f}]", ha="center", fontsize=9)
    # 5 seed 점
    ax.scatter([0]*len(df_seeds), df_seeds["test_pr_auc"], color="black", zorder=3, alpha=0.7, s=20)
    ax.scatter([len(labels)-1]*len(df_rf), df_rf["test_pr_auc"], color="black", zorder=3, alpha=0.7, s=20)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Test PR-AUC")
    ax.set_title(f"S1 Phase 3: Transformer+T2V vs GRU vs GRU-D vs RF ({labeling}, K={K})")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "P3_pr_auc_comparison.png", dpi=130)
    plt.close(fig)

    # 학습 곡선
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for s, l in zip(seeds, train_curves):
        axes[0].plot(l, label=f"seed={s}", alpha=0.7)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("focal loss"); axes[0].set_title("Train loss")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    for s, l in zip(seeds, valid_curves):
        axes[1].plot(l, label=f"seed={s}", alpha=0.7)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("PR-AUC"); axes[1].set_title("Valid PR-AUC")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "P3_train_curves.png", dpi=130)
    plt.close(fig)

    print(f"\n=== Transformer+T2V 5-seed test ===")
    print(f"  PR-AUC mean = {summary['Transformer_T2V']['test_pr_auc_mean']:.4f}  "
          f"std = {summary['Transformer_T2V']['test_pr_auc_std']:.4f}")
    print(f"  Ensemble bootstrap CI = [{bb.ci95_lo:.4f}, {bb.ci95_hi:.4f}]")
    print(f"  P@20 = {summary['Transformer_T2V']['test_p20_mean']:.3f}  "
          f"P@50 = {summary['Transformer_T2V']['test_p50_mean']:.3f}")
    if p1:
        print(f"\n=== Phase 1 GRU 참고: mean={p1['test_pr_auc_mean']:.4f} ± {p1['test_pr_auc_std']:.4f} "
              f"(CI=[{p1['bootstrap_ci_avg_proba']['ci95_lo']:.4f}, {p1['bootstrap_ci_avg_proba']['ci95_hi']:.4f}])")
    if p2:
        print(f"=== Phase 2 GRU-D 참고: mean={p2['test_pr_auc_mean']:.4f} ± {p2['test_pr_auc_std']:.4f} "
              f"(CI=[{p2['bootstrap_ci_avg_proba']['ci95_lo']:.4f}, {p2['bootstrap_ci_avg_proba']['ci95_hi']:.4f}])")
    print(f"=== RF baseline: mean={summary['RF_baseline']['test_pr_auc_mean']:.4f} ± "
          f"{summary['RF_baseline']['test_pr_auc_std']:.4f}")

    print(f"\n[saved] {(OUT_DIR / 'P3_summary.json').relative_to(PROJECT_ROOT)}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeling", default="L3_rolling_H24")
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--model_dim", type=int, default=64)
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--num_layers", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    seeds = (42,) if args.quick else (42, 7, 13, 21, 100)
    epochs = 15 if args.quick else args.epochs
    run_phase3(labeling=args.labeling, K=args.K, seeds=seeds,
                model_dim=args.model_dim, num_heads=args.num_heads,
                num_layers=args.num_layers, epochs=epochs)
