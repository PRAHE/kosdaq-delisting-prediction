"""
S1 Phase 1 — GRU baseline 학습 + S0 RF baseline과 비교.

5 seed × 1 split = 5 측정 + Bayesian bootstrap CI.

출력:
  results/research_s1_phase1/
    P1_test_multi_seed.csv
    P1_summary.json
    P1_pr_auc_comparison.png
    P1_train_curves.png
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from src.research.s0_diagnostic.baseline import PROJECT_ROOT, prepare_baseline, train_baseline_rf
from src.evaluate.bayesian_bootstrap import bayesian_bootstrap, to_dict

OUT_DIR = PROJECT_ROOT / "results" / "research_s1_phase1"


def _top_k_precision(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    if k <= 0:
        return float("nan")
    idx = np.argsort(scores)[::-1][:k]
    return float(np.mean(y_true[idx]))


def run_phase1(
    labeling: str = "L3_rolling_H24",
    K: int = 3,
    seeds=(42, 7, 13, 21, 100),
    hidden_dim: int = 64,
    epochs: int = 30,
    patience: int = 8,
    batch_size: int = 256,
) -> dict:
    from src.models.sequences import prepare_phase1_datasets, summarize
    from src.models.gru import train_gru

    warnings.filterwarnings("ignore")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[S1 Phase 1] labeling={labeling}  K={K}  seeds={list(seeds)}")
    data = prepare_phase1_datasets(labeling=labeling, K=K)
    print("  " + summarize(data["train"], "train"))
    print("  " + summarize(data["valid"], "valid"))
    print("  " + summarize(data["test"],  "test "))

    tr, vl, te = data["train"], data["valid"], data["test"]

    seed_rows = []
    test_probas = []
    train_curves = []
    valid_curves = []

    for seed in seeds:
        print(f"\n--- seed={seed} ---")
        r = train_gru(
            tr.X, tr.M, (tr.y > 0).astype(np.float32),  # binary 변환
            vl.X, vl.M, (vl.y > 0).astype(np.float32),
            te.X, te.M, (te.y > 0).astype(np.float32),
            hidden_dim=hidden_dim,
            epochs=epochs,
            patience=patience,
            batch_size=batch_size,
            seed=seed,
            verbose=False,
        )
        y_test = (te.y > 0).astype(int)
        test_pr = float(average_precision_score(y_test, r.test_proba))
        test_roc = float(roc_auc_score(y_test, r.test_proba))
        p20 = _top_k_precision(y_test, r.test_proba, 20)
        p50 = _top_k_precision(y_test, r.test_proba, 50)
        seed_rows.append({
            "seed": seed,
            "best_epoch": r.best_epoch,
            "valid_pr_auc": r.best_valid_pr_auc,
            "test_pr_auc": test_pr,
            "test_roc_auc": test_roc,
            "test_p20": p20,
            "test_p50": p50,
        })
        test_probas.append(r.test_proba)
        train_curves.append(r.train_losses)
        valid_curves.append(r.valid_pr_aucs)
        print(f"  best_ep={r.best_epoch}  valid_pr={r.best_valid_pr_auc:.4f}  "
              f"test_pr={test_pr:.4f}  test_roc={test_roc:.4f}  P@20={p20:.3f}")

    # 5 seed 평균 probability에 대해 bootstrap CI
    avg_proba = np.mean(np.stack(test_probas), axis=0)
    y_test = (te.y > 0).astype(int)
    bb = bayesian_bootstrap(y_test, avg_proba, metric="pr_auc", n_boot=2000, seed=42)

    # RF baseline 동일 seed에서 비교
    print("\n[ref] RF baseline (S0의 canonical)")
    rf_data = prepare_baseline()
    rf_test_probas = []
    rf_rows = []
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

    # 저장
    import pandas as pd
    df_seeds = pd.DataFrame(seed_rows)
    df_rf = pd.DataFrame(rf_rows)
    df_seeds.to_csv(OUT_DIR / "P1_test_multi_seed.csv", index=False)
    df_rf.to_csv(OUT_DIR / "P1_rf_baseline_multi_seed.csv", index=False)

    summary = {
        "experiment": "S1_Phase1_GRU_baseline",
        "labeling": labeling,
        "K": K,
        "seeds": list(seeds),
        "GRU": {
            "test_pr_auc_mean": float(df_seeds["test_pr_auc"].mean()),
            "test_pr_auc_std":  float(df_seeds["test_pr_auc"].std()),
            "test_pr_auc_min":  float(df_seeds["test_pr_auc"].min()),
            "test_pr_auc_max":  float(df_seeds["test_pr_auc"].max()),
            "test_roc_auc_mean": float(df_seeds["test_roc_auc"].mean()),
            "test_p20_mean":     float(df_seeds["test_p20"].mean()),
            "test_p50_mean":     float(df_seeds["test_p50"].mean()),
            "bootstrap_ci_avg_proba": to_dict(bb),
        },
        "RF_baseline": {
            "test_pr_auc_mean": float(df_rf["test_pr_auc"].mean()),
            "test_pr_auc_std":  float(df_rf["test_pr_auc"].std()),
            "test_roc_auc_mean": float(df_rf["test_roc_auc"].mean()),
            "test_p20_mean":     float(df_rf["test_p20"].mean()),
            "test_p50_mean":     float(df_rf["test_p50"].mean()),
            "bootstrap_ci_avg_proba": to_dict(bb_rf),
        },
    }
    with open(OUT_DIR / "P1_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 그림 1: PR-AUC 비교 (5 seed bar + CI)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = ["GRU (S1 Phase 1)", "RF baseline (S0)"]
    means = [summary["GRU"]["test_pr_auc_mean"], summary["RF_baseline"]["test_pr_auc_mean"]]
    stds = [summary["GRU"]["test_pr_auc_std"], summary["RF_baseline"]["test_pr_auc_std"]]
    cis = [bb, bb_rf]
    err_lo = [m - c.ci95_lo for m, c in zip(means, cis)]
    err_hi = [c.ci95_hi - m for m, c in zip(means, cis)]
    colors = ["#55A868", "#4C72B0"]
    x = np.arange(2)
    ax.bar(x, means, yerr=[err_lo, err_hi], color=colors, capsize=6, edgecolor="k")
    for i, (m, lo, hi) in enumerate(zip(means, [c.ci95_lo for c in cis], [c.ci95_hi for c in cis])):
        ax.text(i, hi + 0.005, f"{m:.4f}\nCI[{lo:.3f},{hi:.3f}]", ha="center", fontsize=9)
    # 5 seed 점도 표시
    ax.scatter([0]*len(df_seeds), df_seeds["test_pr_auc"], color="black", zorder=3, alpha=0.7, s=20)
    ax.scatter([1]*len(df_rf), df_rf["test_pr_auc"], color="black", zorder=3, alpha=0.7, s=20)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Test PR-AUC")
    ax.set_title(f"S1 Phase 1: GRU(K={K}, {labeling}) vs RF baseline")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "P1_pr_auc_comparison.png", dpi=130)
    plt.close(fig)

    # 그림 2: 학습 곡선
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for s, l in zip(seeds, train_curves):
        axes[0].plot(l, label=f"seed={s}", alpha=0.7)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("focal loss")
    axes[0].set_title("Train loss"); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    for s, l in zip(seeds, valid_curves):
        axes[1].plot(l, label=f"seed={s}", alpha=0.7)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("PR-AUC")
    axes[1].set_title("Valid PR-AUC"); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "P1_train_curves.png", dpi=130)
    plt.close(fig)

    print(f"\n=== GRU 5-seed test ===")
    print(f"  PR-AUC mean = {summary['GRU']['test_pr_auc_mean']:.4f}  std = {summary['GRU']['test_pr_auc_std']:.4f}")
    print(f"  Avg-proba bootstrap CI = [{bb.ci95_lo:.4f}, {bb.ci95_hi:.4f}]")
    print(f"  P@20 = {summary['GRU']['test_p20_mean']:.3f}  P@50 = {summary['GRU']['test_p50_mean']:.3f}")
    print(f"\n=== RF baseline ===")
    print(f"  PR-AUC mean = {summary['RF_baseline']['test_pr_auc_mean']:.4f}  std = {summary['RF_baseline']['test_pr_auc_std']:.4f}")
    print(f"  Avg-proba bootstrap CI = [{bb_rf.ci95_lo:.4f}, {bb_rf.ci95_hi:.4f}]")
    print(f"  P@20 = {summary['RF_baseline']['test_p20_mean']:.3f}  P@50 = {summary['RF_baseline']['test_p50_mean']:.3f}")

    print(f"\n[saved] {(OUT_DIR / 'P1_summary.json').relative_to(PROJECT_ROOT)}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeling", default="L3_rolling_H24")
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--quick", action="store_true", help="seed=42만, epochs=15로 빠른 점검")
    args = ap.parse_args()
    seeds = (42,) if args.quick else (42, 7, 13, 21, 100)
    epochs = 15 if args.quick else args.epochs
    run_phase1(labeling=args.labeling, K=args.K, seeds=seeds, epochs=epochs)
