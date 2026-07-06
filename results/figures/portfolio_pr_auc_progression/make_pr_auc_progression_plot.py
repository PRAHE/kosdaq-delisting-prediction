from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager


def set_korean_font() -> None:
    preferred = ["Malgun Gothic", "Noto Sans CJK KR", "Noto Sans KR", "AppleGothic"]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in available:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    set_korean_font()

    out_dir = Path(__file__).resolve().parent
    out_path = out_dir / "pr_auc_progression_line.png"

    stages = [
        "초기 RF\nbaseline",
        "Temporal\nTransformer",
        "Audit signal\n검증",
        "Model-family\ncontrol",
        "최종 모델\nTransformer+Audit",
    ]
    pr_auc = [0.2876, 0.3420, 0.3671, 0.4128, 0.4577]
    notes = [
        "financial\nfeatures",
        "sequence\ntrajectory",
        "RF-38\n+ audit",
        "LogReg-38\nsanity check",
        "market OHLCV\nexcluded",
    ]

    fig, ax = plt.subplots(figsize=(13.6, 7.65), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fbfbfd")

    x = list(range(len(stages)))
    line_color = "#2563eb"
    marker_edge = "#1e3a8a"
    marker_face = "#ffffff"

    ax.plot(
        x,
        pr_auc,
        color=line_color,
        linewidth=3.0,
        marker="o",
        markersize=10,
        markerfacecolor=marker_face,
        markeredgecolor=marker_edge,
        markeredgewidth=2.4,
        zorder=3,
    )

    ax.fill_between(x, pr_auc, [0.24] * len(x), color=line_color, alpha=0.08, zorder=1)

    for idx, (score, note) in enumerate(zip(pr_auc, notes)):
        ax.annotate(
            f"{score:.3f}",
            xy=(idx, score),
            xytext=(0, 18),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
            color="#111827",
        )
        ax.annotate(
            note,
            xy=(idx, score),
            xytext=(0, -34),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=10.5,
            color="#4b5563",
        )

    # Baseline and final reference lines make the progression readable in slides.
    ax.axhline(0.2876, color="#ef4444", linestyle="--", linewidth=1.5, alpha=0.75)
    ax.text(
        4.08,
        0.2876,
        "baseline 0.288",
        va="center",
        ha="left",
        fontsize=10.5,
        color="#b91c1c",
    )
    ax.axhline(0.4577, color="#16a34a", linestyle="--", linewidth=1.5, alpha=0.75)
    ax.text(
        4.08,
        0.4577,
        "final 0.458",
        va="center",
        ha="left",
        fontsize=10.5,
        color="#166534",
    )

    ax.set_title(
        "PR-AUC 개선 흐름: baseline에서 최종 Transformer+Audit 모델까지",
        fontsize=20,
        fontweight="bold",
        pad=18,
    )
    ax.set_ylabel("Test PR-AUC", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=11)
    ax.set_ylim(0.24, 0.50)
    ax.set_xlim(-0.25, len(stages) - 0.45)
    ax.grid(axis="y", color="#c7cdd8", alpha=0.45, linewidth=0.9)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9ca3af")
    ax.spines["bottom"].set_color("#9ca3af")

    ax.text(
        0.0,
        -0.18,
        "해석: 점수 상승 자체보다 temporal modeling, audit signal 검증, market OHLCV artifact 제거를 거쳐 신뢰 가능한 0.458 claim으로 정리한 과정.",
        transform=ax.transAxes,
        fontsize=11.2,
        color="#374151",
        ha="left",
        va="top",
    )

    fig.tight_layout(rect=(0.03, 0.08, 0.98, 0.94))
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
