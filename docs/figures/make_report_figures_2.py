"""최종보고서용 그래프 생성 스크립트 (2) — ★3, ★4.

생성 그래프:
  ★3 PR 곡선 (canonical RF baseline)        -> fig_pr_curve.png
  ★4 Horizon별 PR-AUC 추이 (valid vs test)  -> fig_horizon_trend.png

디자인은 make_report_figures.py(★1·★2)와 동일하게 맞춘다.
  - serif 폰트(HCR Batang 등), 회색조, 검은 테두리, 점선 그리드, dpi=300

★3은 팀 canonical RF baseline(fixed_N1 / exp-A / signed_log1p, PR-AUC 0.2876)을
src.research.s0_diagnostic.baseline 모듈로 재학습해 test 예측확률을 산출한 뒤
precision-recall 곡선을 그린다.

실행:
  .venv\\Scripts\\python.exe docs/figures/make_report_figures_2.py
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(OUT_DIR, "..", ".."))

# ------------------------------------------------------------------
# 폰트 설정 (★1·★2와 동일)
# ------------------------------------------------------------------
_installed = {f.name for f in font_manager.fontManager.ttflist}

for _cand in (
    "HCR Batang",          # 함초롱바탕
    "Helvetica",
    "Times New Roman",
    "NanumMyeongjo",
    "Noto Serif CJK KR",
):
    if _cand in _installed:
        rcParams["font.family"] = _cand
        break

rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "axes.unicode_minus": False,
})


# ------------------------------------------------------------------
# 공통 스타일 (★1·★2와 동일)
# ------------------------------------------------------------------
def apply_report_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(linestyle=":", linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)


# ------------------------------------------------------------------
# ★3 PR 곡선 (canonical RF baseline)
# ------------------------------------------------------------------
def make_pr_curve() -> str:
    import numpy as np
    from sklearn.metrics import average_precision_score, precision_recall_curve

    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from src.research.s0_diagnostic.baseline import (
        prepare_baseline,
        train_baseline_rf,
    )

    # 팀 canonical baseline 재현: fixed_N1 / exp-A / signed_log1p
    data = prepare_baseline(n=1, variant="exp-A", apply_signed_log1p=True)
    rf = train_baseline_rf(data.X_train, data.y_train, seed=42)
    proba = rf.predict_proba(data.X_test)[:, 1]

    y_test = data.y_test
    precision, recall, _ = precision_recall_curve(y_test, proba)
    ap = average_precision_score(y_test, proba)
    base_rate = float(y_test.mean())  # 양성 비율 = 랜덤 분류기 PR 기준선

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.step(recall, precision, where="post", color="0.20", linewidth=1.6,
            label=f"Random Forest (AP = {ap:.3f})")
    ax.axhline(base_rate, color="0.5", linestyle="--", linewidth=0.9,
               label=f"랜덤 baseline (양성률 {base_rate:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("PR 곡선 — Random Forest baseline (test 2024)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")

    apply_report_style(ax)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig_pr_curve.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  (PR-AUC/AP = {ap:.4f}, test 양성률 = {base_rate:.4f})")
    return path


# ------------------------------------------------------------------
# ★4 Horizon별 PR-AUC 추이 (RF, valid vs test)
# ------------------------------------------------------------------
def make_horizon_trend() -> str:
    # 결과물 (4) Horizon Sweep — RF PR-AUC (H10~H24)
    horizons = [10, 12, 14, 16, 18, 20, 22, 24]
    valid = [0.181, 0.238, 0.208, 0.237, 0.305, 0.331, 0.381, 0.391]
    test = [0.174, 0.236, 0.193, 0.256, 0.235, 0.264, 0.261, 0.256]

    peak = max(range(len(test)), key=lambda i: test[i])  # test peak = H20

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(horizons, test, color="0.20", linewidth=1.8, marker="o",
            markersize=5, label="Test (2024)")
    ax.plot(horizons, valid, color="0.55", linewidth=1.4, marker="s",
            markersize=4, linestyle="--", label="Valid (2023)")

    # test peak 강조
    ax.scatter([horizons[peak]], [test[peak]], s=90, facecolors="none",
               edgecolors="black", linewidths=1.4, zorder=5)
    ax.annotate(f"Test peak\nH{horizons[peak]} = {test[peak]:.3f}",
                xy=(horizons[peak], test[peak]),
                xytext=(horizons[peak] - 5.5, test[peak] - 0.07),
                fontsize=9,
                arrowprops=dict(arrowstyle="->", color="0.3", linewidth=0.8))

    ax.set_xlabel("Horizon (개월)")
    ax.set_ylabel("PR-AUC")
    ax.set_title("Horizon별 PR-AUC 추이 (Random Forest)")
    ax.set_xticks(horizons)
    ax.set_ylim(0, max(valid) * 1.18)
    ax.legend(loc="upper left")

    apply_report_style(ax)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig_horizon_trend.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("saved:", make_horizon_trend())
    print("saved:", make_pr_curve())
