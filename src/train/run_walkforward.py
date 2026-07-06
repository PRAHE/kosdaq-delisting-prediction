"""
S1 Walk-Forward 재검증.

S0 walk-forward CV 구조와 동일하게 4 fold × N seed로 GRU(P1)와 Transformer+T2V(P3)
를 측정한다. S0-A2에서 측정된 RF baseline의 fold별 결과(`results/research_s0_diagnostic/
A2_walk_forward_results.csv`)와 함께 3-way 비교.

출력:
  results/research_s1_walkforward/
    WF_grid.csv           # 모델 × fold × seed 측정
    WF_fold_summary.csv   # fold별 mean ± std 비교
    WF_summary.json
    WF_comparison.png     # fold heatmap + 평균 + std bar
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
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.research.s0_diagnostic.baseline import PROJECT_ROOT

from src.models.gru import train_gru
from src.train.run_phase3_transformer import train_transformer
from src.models.sequences import prepare_fold_datasets, summarize

_OUT_BASE = PROJECT_ROOT / "results" / "research_s1_walkforward"


def _resolve_out_dir(with_market: bool, with_audit: bool = False, with_supervision: bool = False) -> Path:
    flags = []
    if with_market: flags.append("market")
    if with_audit: flags.append("audit")
    if with_supervision: flags.append("supervision")
    if not flags:
        return _OUT_BASE
    return PROJECT_ROOT / "results" / ("research_s1_walkforward_" + "_".join(flags))
FOLDS = [
    (1, (2015, 2019), 2020),
    (2, (2015, 2020), 2021),
    (3, (2015, 2021), 2022),
    (4, (2015, 2022), 2023),
]


def _top_k(y_true, scores, k):
    if k <= 0:
        return float("nan")
    idx = np.argsort(scores)[::-1][:k]
    return float(np.mean(y_true[idx]))


def _eval(y_true, proba):
    return {
        "pr_auc": float(average_precision_score(y_true, proba)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "p20": _top_k(y_true, proba, 20),
        "p50": _top_k(y_true, proba, 50),
    }


def run(
    seeds=(42, 7, 13),
    labeling="L3_rolling_H24",
    K=3,
    epochs=30,
    with_market: bool = False,
    with_audit: bool = False,
    with_supervision: bool = False,
) -> dict:
    warnings.filterwarnings("ignore")
    out_dir = _resolve_out_dir(with_market, with_audit, with_supervision)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    total = len(FOLDS) * len(seeds) * 2
    cnt = 0
    flags = []
    if with_market: flags.append("+market")
    if with_audit: flags.append("+audit")
    if with_supervision: flags.append("+supervision")
    flag_s = " ".join(flags)
    for fold_id, (yr_lo, yr_hi), valid_year in FOLDS:
        print(f"\n=== Fold {fold_id}: train {yr_lo}-{yr_hi}, valid {valid_year} {flag_s} ===")
        d = prepare_fold_datasets(fold_id, (yr_lo, yr_hi), valid_year,
                                    labeling=labeling, K=K,
                                    with_market=with_market, with_audit=with_audit,
                                    with_supervision=with_supervision)
        print(f"  features: {len(d['feature_cols'])} cols")
        print("  " + summarize(d["train"], "train"))
        print("  " + summarize(d["valid"], "valid"))
        y_valid = (d["valid"].y > 0).astype(int)
        y_test  = (d["test"].y  > 0).astype(int)

        for seed in seeds:
            # ---------- GRU ----------
            cnt += 1
            print(f"  [{cnt}/{total}] GRU fold={fold_id} seed={seed}...", flush=True)
            r_gru = train_gru(
                d["train"].X, d["train"].M, (d["train"].y > 0).astype(np.float32),
                d["valid"].X, d["valid"].M, y_valid.astype(np.float32),
                d["test"].X,  d["test"].M,  y_test.astype(np.float32),
                hidden_dim=64, epochs=epochs, patience=8, batch_size=256,
                seed=seed, verbose=False,
            )
            v = _eval(y_valid, r_gru.valid_proba)
            t = _eval(y_test,  r_gru.test_proba)
            rows.append({
                "model": "GRU",
                "fold": fold_id,
                "train_years": f"{yr_lo}-{yr_hi}",
                "valid_year": valid_year,
                "seed": seed,
                "best_epoch": r_gru.best_epoch,
                "valid_pr_auc": v["pr_auc"], "valid_roc_auc": v["roc_auc"],
                "test_pr_auc":  t["pr_auc"], "test_roc_auc":  t["roc_auc"],
                "test_p20": t["p20"], "test_p50": t["p50"],
            })
            print(f"    GRU best_ep={r_gru.best_epoch} valid_pr={v['pr_auc']:.4f} test_pr={t['pr_auc']:.4f}")

            # ---------- Transformer + T2V ----------
            cnt += 1
            print(f"  [{cnt}/{total}] Transformer+T2V fold={fold_id} seed={seed}...", flush=True)
            r_tx = train_transformer(
                d["train"], d["valid"], d["test"],
                model_dim=64, num_heads=4, num_layers=1, dropout=0.2,
                epochs=epochs, patience=8, batch_size=256, seed=seed,
                lr=5e-4, verbose=False,
            )
            v = _eval(y_valid, _proba_from_transformer_result(r_tx, "valid", y_valid))
            # train_transformer는 test_proba만 저장하므로 valid는 별도 평가가 필요
            # 그러나 r_tx.valid_pr_aucs[best_ep-1]이 valid 정점값이므로 그것을 사용
            v_pr = r_tx.best_valid_pr_auc
            t = _eval(y_test, r_tx.test_proba)
            rows.append({
                "model": "Transformer+T2V",
                "fold": fold_id,
                "train_years": f"{yr_lo}-{yr_hi}",
                "valid_year": valid_year,
                "seed": seed,
                "best_epoch": r_tx.best_epoch,
                "valid_pr_auc": v_pr, "valid_roc_auc": float("nan"),
                "test_pr_auc":  t["pr_auc"], "test_roc_auc":  t["roc_auc"],
                "test_p20": t["p20"], "test_p50": t["p50"],
            })
            print(f"    TX  best_ep={r_tx.best_epoch} valid_pr={v_pr:.4f} test_pr={t['pr_auc']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "WF_grid.csv", index=False)

    # RF (S0-A2 결과 로드)
    rf_path = PROJECT_ROOT / "results" / "research_s0_diagnostic" / "A2_walk_forward_results.csv"
    if rf_path.exists():
        rf_df = pd.read_csv(rf_path)
        rf_df = rf_df.assign(model="RF")
        rf_test_df = pd.read_csv(PROJECT_ROOT / "results" / "research_s0_diagnostic" / "A2_test_multi_seed.csv")
    else:
        rf_df = pd.DataFrame()
        rf_test_df = pd.DataFrame()

    # fold별 summary (mean, std across seeds)
    summary_rows = []
    for model in ["GRU", "Transformer+T2V"]:
        for fold_id in range(1, 5):
            sub = df[(df["model"] == model) & (df["fold"] == fold_id)]
            if len(sub) == 0:
                continue
            summary_rows.append({
                "model": model,
                "fold": fold_id,
                "valid_year": sub["valid_year"].iloc[0],
                "n_seeds": len(sub),
                "valid_pr_mean": float(sub["valid_pr_auc"].mean()),
                "valid_pr_std":  float(sub["valid_pr_auc"].std()),
                "test_pr_mean": float(sub["test_pr_auc"].mean()),
                "test_pr_std":  float(sub["test_pr_auc"].std()),
                "test_roc_mean": float(sub["test_roc_auc"].mean()),
                "test_p20_mean": float(sub["test_p20"].mean()),
                "test_p50_mean": float(sub["test_p50"].mean()),
            })
    # RF 추가 (fold별)
    if not rf_df.empty:
        for fold_id in range(1, 5):
            sub = rf_df[rf_df["fold"] == fold_id]
            summary_rows.append({
                "model": "RF",
                "fold": fold_id,
                "valid_year": int(sub["valid_year"].iloc[0]) if "valid_year" in sub.columns else 2019 + fold_id,
                "n_seeds": len(sub),
                "valid_pr_mean": float(sub["pr_auc"].mean()),
                "valid_pr_std":  float(sub["pr_auc"].std()),
                "test_pr_mean": float("nan"),  # RF의 fold별 test는 따로 측정 안 함
                "test_pr_std":  float("nan"),
                "test_roc_mean": float(sub["roc_auc"].mean()),
                "test_p20_mean": float("nan"),
                "test_p50_mean": float("nan"),
            })
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(out_dir / "WF_fold_summary.csv", index=False)

    # 전체 평균
    overall = (
        df.groupby("model")[["valid_pr_auc", "test_pr_auc", "test_roc_auc", "test_p20", "test_p50"]]
        .agg(["mean", "std"])
        .round(4)
    )

    summary_json = {
        "experiment": "S1_walkforward",
        "labeling": labeling,
        "K": K,
        "seeds": list(seeds),
        "folds": [{"fold": f, "train": f"{yr_lo}-{yr_hi}", "valid": vy}
                  for (f, (yr_lo, yr_hi), vy) in FOLDS],
        "overall_mean": {
            model: {
                "valid_pr_mean": float(df[df["model"] == model]["valid_pr_auc"].mean()),
                "test_pr_mean":  float(df[df["model"] == model]["test_pr_auc"].mean()),
                "test_pr_std_across_all": float(df[df["model"] == model]["test_pr_auc"].std()),
                "test_roc_mean": float(df[df["model"] == model]["test_roc_auc"].mean()),
                "test_p20_mean": float(df[df["model"] == model]["test_p20"].mean()),
                "test_p50_mean": float(df[df["model"] == model]["test_p50"].mean()),
            }
            for model in ["GRU", "Transformer+T2V"]
        },
        "fold_summary": df_summary.to_dict(orient="records"),
    }
    with open(out_dir / "WF_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2)

    # 그림: fold별 valid PR-AUC (model × fold)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # left: valid PR-AUC fold별 (3 모델)
    fold_x = [1, 2, 3, 4]
    valid_years = [2020, 2021, 2022, 2023]
    width = 0.25
    for i, (model, color) in enumerate([
        ("GRU", "#55A868"),
        ("Transformer+T2V", "#9B59B6"),
        ("RF", "#4C72B0"),
    ]):
        sub = df_summary[df_summary["model"] == model].sort_values("fold")
        if len(sub) == 0:
            continue
        x = np.array(fold_x) + (i - 1) * width
        axes[0].bar(x, sub["valid_pr_mean"].values, width, yerr=sub["valid_pr_std"].values,
                     color=color, capsize=3, label=model, edgecolor="k")
    axes[0].set_xticks(fold_x)
    axes[0].set_xticklabels([f"Fold{f}\nvalid={vy}" for f, vy in zip(fold_x, valid_years)])
    axes[0].set_ylabel("Valid PR-AUC (mean ± std)")
    axes[0].set_title("Walk-forward Valid PR-AUC per fold")
    axes[0].legend()
    axes[0].grid(alpha=0.3, axis="y")

    # right: test PR-AUC fold별 (GRU, Transformer만)
    for i, (model, color) in enumerate([
        ("GRU", "#55A868"),
        ("Transformer+T2V", "#9B59B6"),
    ]):
        sub = df_summary[df_summary["model"] == model].sort_values("fold")
        if len(sub) == 0 or sub["test_pr_mean"].isna().all():
            continue
        x = np.array(fold_x) + (i - 0.5) * width
        axes[1].bar(x, sub["test_pr_mean"].values, width, yerr=sub["test_pr_std"].values,
                     color=color, capsize=3, label=model, edgecolor="k")
    axes[1].set_xticks(fold_x)
    axes[1].set_xticklabels([f"Fold{f}\nvalid={vy}" for f, vy in zip(fold_x, valid_years)])
    axes[1].set_ylabel("Test 2024 PR-AUC (mean ± std)")
    axes[1].set_title("Walk-forward Test PR-AUC per fold (model trained on each fold)")
    axes[1].legend()
    axes[1].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(out_dir / "WF_comparison.png", dpi=130)
    plt.close(fig)

    print("\n=== Walk-forward summary (mean ± std across folds) ===")
    print(overall.to_string())
    print(f"\n[saved] {(out_dir / 'WF_grid.csv').relative_to(PROJECT_ROOT)}")
    print(f"[saved] {(out_dir / 'WF_fold_summary.csv').relative_to(PROJECT_ROOT)}")
    print(f"[saved] {(out_dir / 'WF_summary.json').relative_to(PROJECT_ROOT)}")
    print(f"[saved] {(out_dir / 'WF_comparison.png').relative_to(PROJECT_ROOT)}")
    return summary_json


def _proba_from_transformer_result(r, split, y):
    # train_transformer is from .run_s1_phase3 — only stores test_proba.
    # For valid, we reuse best_valid_pr_auc already.
    # Return placeholder so _eval is not called for transformer valid.
    return np.zeros(len(y), dtype=float)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,7,13")
    ap.add_argument("--labeling", default="L3_rolling_H24")
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--with-market", action="store_true",
                     help="시장 피처 6개 추가")
    ap.add_argument("--with-audit", action="store_true",
                     help="감사의견 피처 5개 추가")
    ap.add_argument("--with-supervision", action="store_true",
                     help="관리종목 피처 6개 추가")
    args = ap.parse_args()
    seeds = tuple(int(s) for s in args.seeds.split(","))
    run(seeds=seeds, labeling=args.labeling, K=args.K, epochs=args.epochs,
         with_market=args.with_market, with_audit=args.with_audit,
         with_supervision=args.with_supervision)
