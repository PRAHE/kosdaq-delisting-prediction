"""
Best baseline의 Bayesian bootstrap CI + S0 baseline과 통계 비교.

핵심 질문:
  "Transformer + 시장 + 감사 (44 features)의 walk-forward 평균 PR-AUC 0.408이
   S0 baseline CI [0.180, 0.421]을 통계적으로 넘었는가?"

방법:
  1) 단일 split (train 2015~2022 / valid 2023 / test 2024) — baseline과 동일 조건
  2) Transformer 44 features를 5 seed로 학습 → 5-seed test probability 평균
  3) RF baseline (S0 canonical) 5 seed로 재학습 → 5-seed probability 평균
  4) Paired Bayesian bootstrap (n_boot=5000):
     같은 Dirichlet 가중치로 두 모델의 PR-AUC posterior 동시 산출
  5) Δ = best - baseline posterior
  6) P(best > baseline) 추정

결정 기준:
  P(best > baseline) > 0.975 → 통계적 천장 돌파 (단측 유의수준 2.5%)
  Δ 95% CI가 양수 영역에만 있음 → 동일 결론 (양측 5%)
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.research.s0_diagnostic.baseline import (
    PROJECT_ROOT,
    prepare_baseline,
    train_baseline_rf,
)
from src.evaluate.bayesian_bootstrap import bayesian_bootstrap
from src.train.run_phase3_transformer import train_transformer
from src.models.sequences import prepare_phase1_datasets

OUT_DIR = PROJECT_ROOT / "results" / "research_best_ci"
N_BOOT = 5000
SEEDS = [42, 7, 13, 21, 100]


def main() -> None:
    warnings.filterwarnings("ignore")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # === 1) Best 모델: Transformer + 시장 + 감사, 단일 split, 5 seed ===
    print("[1/3] Best baseline (Transformer 44 features, single split, 5 seeds)")
    data = prepare_phase1_datasets(
        labeling="L3_rolling_H24", K=3,
        with_market=True, with_audit=True, with_supervision=False,
    )
    tr, vl, te = data["train"], data["valid"], data["test"]
    print(f"  features: {len(data['feature_cols'])} cols")
    print(f"  train N={len(tr.y)}, pos={int((tr.y>0).sum())}")
    print(f"  test  N={len(te.y)}, pos={int((te.y>0).sum())}")

    best_probas = []
    best_seed_pr = []
    for seed in SEEDS:
        r = train_transformer(tr, vl, te,
                                model_dim=64, num_heads=4, num_layers=1,
                                dropout=0.2, epochs=25, patience=8,
                                batch_size=256, seed=seed, lr=5e-4,
                                verbose=False)
        y_t = (te.y > 0).astype(int)
        pr = float(average_precision_score(y_t, r.test_proba))
        best_probas.append(r.test_proba)
        best_seed_pr.append(pr)
        print(f"  seed={seed:>3}: test PR-AUC={pr:.4f}  best_ep={r.best_epoch}")
    best_avg = np.mean(np.stack(best_probas), axis=0)
    y_test_best = (te.y > 0).astype(int)
    best_point = float(average_precision_score(y_test_best, best_avg))
    print(f"\n  5-seed mean PR-AUC = {np.mean(best_seed_pr):.4f} ± {np.std(best_seed_pr):.4f}")
    print(f"  Ensemble probability PR-AUC = {best_point:.4f}")

    # === 2) S0 baseline RF: 단일 split, 5 seed (재학습) ===
    print("\n[2/3] S0 baseline RF (33 features, single split, 5 seeds)")
    rf_data = prepare_baseline()
    rf_probas = []
    rf_seed_pr = []
    for seed in SEEDS:
        rf = train_baseline_rf(rf_data.X_train, rf_data.y_train, seed=seed)
        p = rf.predict_proba(rf_data.X_test)[:, 1]
        rf_probas.append(p)
        pr = float(average_precision_score(rf_data.y_test, p))
        rf_seed_pr.append(pr)
        print(f"  seed={seed:>3}: test PR-AUC={pr:.4f}")
    rf_avg = np.mean(np.stack(rf_probas), axis=0)
    rf_point = float(average_precision_score(rf_data.y_test, rf_avg))
    print(f"\n  5-seed mean PR-AUC = {np.mean(rf_seed_pr):.4f} ± {np.std(rf_seed_pr):.4f}")
    print(f"  Ensemble probability PR-AUC = {rf_point:.4f}")

    # === 3) Paired Bayesian bootstrap ===
    # baseline과 best는 같은 test 행 인덱스를 공유한다 (둘 다 fixed_v1 test).
    # 따라서 같은 Dirichlet 가중치로 두 모델 동시에 평가 가능.
    print(f"\n[3/3] Paired Bayesian bootstrap (n={N_BOOT})")
    n = len(y_test_best)
    assert len(rf_data.y_test) == n, "test set 크기 불일치!"
    # y_test 같은지 확인
    np.testing.assert_array_equal(y_test_best, rf_data.y_test.astype(int),
                                     err_msg="test labels 불일치!")

    rng = np.random.default_rng(42)
    gammas = rng.standard_gamma(shape=1.0, size=(N_BOOT, n))
    weights = gammas / gammas.sum(axis=1, keepdims=True) * n

    best_post = np.empty(N_BOOT)
    base_post = np.empty(N_BOOT)
    for i in range(N_BOOT):
        best_post[i] = float(average_precision_score(y_test_best, best_avg, sample_weight=weights[i]))
        base_post[i] = float(average_precision_score(rf_data.y_test, rf_avg, sample_weight=weights[i]))

    delta = best_post - base_post
    p_best_gt = float(np.mean(best_post > base_post))
    p_best_geq = float(np.mean(best_post >= base_post))

    best_ci_lo = float(np.percentile(best_post, 2.5))
    best_ci_hi = float(np.percentile(best_post, 97.5))
    base_ci_lo = float(np.percentile(base_post, 2.5))
    base_ci_hi = float(np.percentile(base_post, 97.5))
    d_ci_lo = float(np.percentile(delta, 2.5))
    d_ci_hi = float(np.percentile(delta, 97.5))

    print(f"  Best CI95   = [{best_ci_lo:.4f}, {best_ci_hi:.4f}], point={best_point:.4f}")
    print(f"  Base CI95   = [{base_ci_lo:.4f}, {base_ci_hi:.4f}], point={rf_point:.4f}")
    print(f"  Δ mean      = {delta.mean():.4f}")
    print(f"  Δ CI95      = [{d_ci_lo:.4f}, {d_ci_hi:.4f}]")
    print(f"  P(best > baseline) = {p_best_gt:.4f}")
    if d_ci_lo > 0:
        print("  → Δ 95% CI 전부 양수 → 통계적으로 best > baseline 입증")
    else:
        print("  → Δ 95% CI가 0을 포함 → 통계적 차이 미확정")

    # === Save ===
    summary = {
        "experiment": "best_baseline_vs_S0_baseline_paired_bootstrap",
        "n_boot": N_BOOT,
        "seeds": SEEDS,
        "best_model": {
            "name": "Transformer+T2V + market + audit (44 features)",
            "5seed_test_pr_auc": best_seed_pr,
            "5seed_mean": float(np.mean(best_seed_pr)),
            "5seed_std": float(np.std(best_seed_pr)),
            "ensemble_point": best_point,
            "ensemble_ci95": [best_ci_lo, best_ci_hi],
        },
        "baseline_model": {
            "name": "RF baseline (S0 canonical, 33 features)",
            "5seed_test_pr_auc": rf_seed_pr,
            "5seed_mean": float(np.mean(rf_seed_pr)),
            "5seed_std": float(np.std(rf_seed_pr)),
            "ensemble_point": rf_point,
            "ensemble_ci95": [base_ci_lo, base_ci_hi],
        },
        "paired_delta": {
            "mean": float(delta.mean()),
            "ci95_lo": d_ci_lo,
            "ci95_hi": d_ci_hi,
            "p_best_gt_baseline": p_best_gt,
            "p_best_geq_baseline": p_best_geq,
            "ci95_strictly_positive": bool(d_ci_lo > 0),
        },
    }
    with open(OUT_DIR / "best_ci_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # CSV
    pd.DataFrame({"seed": SEEDS, "best_pr_auc": best_seed_pr, "baseline_pr_auc": rf_seed_pr}).to_csv(
        OUT_DIR / "seed_results.csv", index=False,
    )

    # === Plot ===
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # left: two posteriors overlay
    bins = np.linspace(min(base_post.min(), best_post.min()) - 0.01,
                       max(base_post.max(), best_post.max()) + 0.01, 60)
    axes[0].hist(base_post, bins=bins, alpha=0.55, color="#4C72B0", density=True,
                  label=f"S0 RF baseline\n  point={rf_point:.3f}, CI=[{base_ci_lo:.3f},{base_ci_hi:.3f}]")
    axes[0].hist(best_post, bins=bins, alpha=0.65, color="#9B59B6", density=True,
                  label=f"Transformer+market+audit\n  point={best_point:.3f}, CI=[{best_ci_lo:.3f},{best_ci_hi:.3f}]")
    axes[0].axvline(rf_point, color="#4C72B0", linestyle="--", linewidth=1, alpha=0.6)
    axes[0].axvline(best_point, color="#9B59B6", linestyle="--", linewidth=1, alpha=0.6)
    axes[0].set_xlabel("Test PR-AUC")
    axes[0].set_ylabel("density")
    axes[0].set_title(f"두 모델의 PR-AUC posterior (paired Bayesian bootstrap n={N_BOOT})")
    axes[0].legend(fontsize=9, loc="upper left")
    axes[0].grid(alpha=0.3)

    # right: Δ posterior
    axes[1].hist(delta, bins=60, color="#55A868", density=True, edgecolor="white")
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1.5, label="Δ = 0")
    axes[1].axvline(delta.mean(), color="black", linestyle="-", linewidth=1.5,
                     label=f"Δ mean = {delta.mean():.4f}")
    axes[1].axvspan(d_ci_lo, d_ci_hi, color="#55A868", alpha=0.18,
                     label=f"95% CI [{d_ci_lo:.3f}, {d_ci_hi:.3f}]")
    axes[1].set_xlabel("Δ = best PR-AUC − baseline PR-AUC")
    axes[1].set_ylabel("density")
    axes[1].set_title(f"Paired Δ posterior  (P(best>base)={p_best_gt:.3f})")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "best_ci_comparison.png", dpi=130)
    plt.close(fig)

    print(f"\n[saved] {(OUT_DIR / 'best_ci_summary.json').relative_to(PROJECT_ROOT)}")
    print(f"[saved] {(OUT_DIR / 'best_ci_comparison.png').relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
