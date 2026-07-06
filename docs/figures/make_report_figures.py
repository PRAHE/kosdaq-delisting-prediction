"""최종보고서용 그래프 생성 스크립트.

생성 그래프:
  ★1 모델별 Test PR-AUC 막대그래프      -> fig_model_pr_auc.png
  ★2 핵심 부실 신호 특징 중요도 가로막대 -> fig_feature_importance.png

수치는 docs/최종보고서.md 결과물 (9)·(10) 표와 동일하다.
실행:
  .venv\\Scripts\\python.exe docs/figures/make_report_figures.py
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------
# 폰트 설정
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
# 공통 스타일
# ------------------------------------------------------------------
def apply_report_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(
        linestyle=":",
        linewidth=0.6,
        alpha=0.6,
    )

    ax.set_axisbelow(True)


# ------------------------------------------------------------------
# 1. 모델별 Test PR-AUC
# ------------------------------------------------------------------
def make_model_pr_auc() -> str:

    models = [
        ("Random Forest", 0.287),
        ("Logistic Regression", 0.271),
        ("LightGBM", 0.263),
        ("XGBoost", 0.231),
        ("Gradient Boosting", 0.112),
    ]

    names = [m[0] for m in models]
    scores = [m[1] for m in models]

    best_idx = max(range(len(scores)), key=lambda i: scores[i])

    colors = ["0.70"] * len(scores)
    colors[best_idx] = "0.30"

    fig, ax = plt.subplots(figsize=(7, 4))

    bars = ax.bar(
        names,
        scores,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
        width=0.65,
    )

    ax.set_ylabel("Test PR-AUC")
    ax.set_title("모델별 Test PR-AUC")

    ax.set_ylim(0, max(scores) * 1.20)

    # Random baseline
    ax.axhline(
        0.008,
        color="0.4",
        linestyle="--",
        linewidth=0.8,
    )

    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            score + 0.004,
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.xticks(rotation=15, ha="right")

    apply_report_style(ax)

    fig.tight_layout()

    path = os.path.join(OUT_DIR, "fig_model_pr_auc.png")

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    return path


# ------------------------------------------------------------------
# 2. Feature Importance
# ------------------------------------------------------------------
def make_feature_importance() -> str:

    features = [
        ("총자본순이익률 (ROA)", 0.090),
        ("유보액/납입자본비율", 0.081),
        ("총자본영업이익률", 0.071),
        ("총자산증가율", 0.048),
        ("부채비율", 0.039),
        ("차입금의존도", 0.034),
    ]

    names = [f[0] for f in features]
    values = [f[1] for f in features]

    # 중요도 순위에 따라 회색조
    colors = [
        "0.25",
        "0.35",
        "0.45",
        "0.60",
        "0.70",
        "0.80",
    ]

    fig, ax = plt.subplots(figsize=(7, 4))

    y_pos = range(len(names))

    bars = ax.barh(
        y_pos,
        values,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        height=0.65,
    )

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(names)

    ax.invert_yaxis()

    ax.set_xlabel("Feature Importance")
    ax.set_title("핵심 부실 신호 특징 중요도")

    ax.set_xlim(0, max(values) * 1.18)

    for y, val in zip(y_pos, values):
        ax.text(
            val + 0.002,
            y,
            f"{val:.3f}",
            va="center",
            ha="left",
            fontsize=9,
        )

    apply_report_style(ax)

    fig.tight_layout()

    path = os.path.join(
        OUT_DIR,
        "fig_feature_importance.png"
    )

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    return path


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
if __name__ == "__main__":

    p1 = make_model_pr_auc()
    p2 = make_feature_importance()

    print("saved:", p1)
    print("saved:", p2)