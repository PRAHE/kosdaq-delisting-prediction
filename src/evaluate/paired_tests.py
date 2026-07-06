"""
Paired statistical tests for ranking metrics.

지원 검정:
  - paired_wilcoxon: scipy.stats.wilcoxon. fold-level paired PR-AUC 비교용.
  - delong_test: paired ROC-AUC 비교 (Sun & Xu 2014의 fast algorithm).
  - mcnemar: 동일 sample에 대한 두 분류기의 confusion 차이.
  - holm_bonferroni: 다중검정 보정.
  - bayesian_compare: baseline의 bootstrap posterior를 활용한
    "P(baseline ≥ exp)" 베이지안 비교 (probas 없어도 사용 가능).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# 1) Wilcoxon
# ---------------------------------------------------------------------------


@dataclass
class PairedTestResult:
    name: str
    statistic: float
    pvalue: float
    delta: float
    extra: dict


def paired_wilcoxon(a: np.ndarray, b: np.ndarray) -> PairedTestResult:
    """Wilcoxon signed-rank test (paired).

    a, b: 동일 길이의 paired metric 시퀀스 (예: fold-level PR-AUC).
    H0: median(a - b) == 0.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diffs = a - b
    # 모든 차이가 0이면 wilcoxon이 에러를 내므로 처리
    if np.allclose(diffs, 0):
        return PairedTestResult(
            name="wilcoxon",
            statistic=0.0,
            pvalue=1.0,
            delta=0.0,
            extra={"diffs": diffs.tolist(), "n": len(diffs)},
        )
    res = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    return PairedTestResult(
        name="wilcoxon",
        statistic=float(res.statistic),
        pvalue=float(res.pvalue),
        delta=float(np.median(diffs)),
        extra={"mean_diff": float(np.mean(diffs)), "n": len(diffs)},
    )


# ---------------------------------------------------------------------------
# 2) DeLong test for paired ROC-AUC (Sun & Xu 2014 fast algorithm)
# ---------------------------------------------------------------------------


def _midrank(x: np.ndarray) -> np.ndarray:
    """midrank (Sun & Xu의 algorithm 1)."""
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    T = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T_unordered = np.empty(n)
    T_unordered[order] = T
    return T_unordered


def _delong_aucs_and_covariance(y_true: np.ndarray, probas: np.ndarray):
    """fast DeLong algorithm. probas shape (k, n) for k classifiers."""
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    m = len(pos_idx)
    n = len(neg_idx)
    k = probas.shape[0]

    pos = probas[:, pos_idx]
    neg = probas[:, neg_idx]

    tx = np.empty((k, m))
    ty = np.empty((k, n))
    tz = np.empty((k, m + n))
    for r in range(k):
        tx[r] = _midrank(pos[r])
        ty[r] = _midrank(neg[r])
        tz[r] = _midrank(np.concatenate([pos[r], neg[r]]))

    aucs = (tz[:, :m].sum(axis=1) / m - (m + 1) / 2.0) / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    if k == 1:
        sx = np.array([[sx]])
        sy = np.array([[sy]])
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_test(y_true: np.ndarray, proba_a: np.ndarray, proba_b: np.ndarray) -> PairedTestResult:
    """Paired DeLong test for AUC difference (two-sided)."""
    y_true = np.asarray(y_true)
    probas = np.vstack([np.asarray(proba_a), np.asarray(proba_b)])
    aucs, cov = _delong_aucs_and_covariance(y_true, probas)
    L = np.array([1.0, -1.0])
    diff = aucs[0] - aucs[1]
    var = float(L @ cov @ L.T)
    if var <= 0:
        return PairedTestResult(
            name="delong",
            statistic=0.0,
            pvalue=1.0,
            delta=float(diff),
            extra={"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "var": var},
        )
    z = diff / np.sqrt(var)
    pvalue = 2 * (1 - stats.norm.cdf(abs(z)))
    return PairedTestResult(
        name="delong",
        statistic=float(z),
        pvalue=float(pvalue),
        delta=float(diff),
        extra={"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "var": var},
    )


# ---------------------------------------------------------------------------
# 3) McNemar test
# ---------------------------------------------------------------------------


def mcnemar(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> PairedTestResult:
    """McNemar's chi-squared test with continuity correction."""
    y_true = np.asarray(y_true).astype(bool)
    pa = np.asarray(pred_a).astype(bool)
    pb = np.asarray(pred_b).astype(bool)
    # b = a correct, b wrong;  c = a wrong, b correct
    b = int(np.sum((pa == y_true) & (pb != y_true)))
    c = int(np.sum((pa != y_true) & (pb == y_true)))
    if b + c == 0:
        return PairedTestResult(
            name="mcnemar",
            statistic=0.0,
            pvalue=1.0,
            delta=0.0,
            extra={"b": b, "c": c},
        )
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    pvalue = 1 - stats.chi2.cdf(chi2, df=1)
    return PairedTestResult(
        name="mcnemar",
        statistic=float(chi2),
        pvalue=float(pvalue),
        delta=float(b - c),
        extra={"b_only_a_correct": b, "c_only_b_correct": c},
    )


# ---------------------------------------------------------------------------
# 4) Holm-Bonferroni
# ---------------------------------------------------------------------------


def holm_bonferroni(pvalues: list[float]) -> list[float]:
    """Holm step-down adjusted p-values."""
    p_arr = np.asarray(pvalues, dtype=float)
    n = len(p_arr)
    order = np.argsort(p_arr)
    adjusted = np.empty(n)
    max_so_far = 0.0
    for rank, idx in enumerate(order):
        adj = (n - rank) * p_arr[idx]
        adj = min(adj, 1.0)
        max_so_far = max(max_so_far, adj)
        adjusted[idx] = max_so_far
    return adjusted.tolist()


# ---------------------------------------------------------------------------
# 5) Bayesian posterior comparison
# ---------------------------------------------------------------------------


@dataclass
class BayesianCompareResult:
    exp_id: str
    exp_value: float
    baseline_point: float
    delta: float
    in_ci95: bool
    ci95_lo: float
    ci95_hi: float
    p_baseline_geq_exp: float  # baseline의 posterior에서 baseline >= exp일 비율
    p_baseline_leq_exp: float


def bayesian_compare(
    exp_id: str,
    exp_value: float,
    baseline_posterior: np.ndarray,
    baseline_point: float,
) -> BayesianCompareResult:
    """baseline의 bootstrap posterior를 활용해 exp의 위치를 측정.

    p_baseline_geq_exp = P(baseline_PR_AUC >= exp_PR_AUC | baseline 데이터)
        값이 작다 → exp가 baseline보다 유의하게 우수
    p_baseline_leq_exp = 1 - above
        값이 작다 → exp가 baseline보다 유의하게 열등 (대부분의 exp_010~018가 해당)
    """
    bp = np.asarray(baseline_posterior)
    lo, hi = np.percentile(bp, 2.5), np.percentile(bp, 97.5)
    return BayesianCompareResult(
        exp_id=exp_id,
        exp_value=float(exp_value),
        baseline_point=float(baseline_point),
        delta=float(exp_value - baseline_point),
        in_ci95=bool(lo <= exp_value <= hi),
        ci95_lo=float(lo),
        ci95_hi=float(hi),
        p_baseline_geq_exp=float(np.mean(bp >= exp_value)),
        p_baseline_leq_exp=float(np.mean(bp <= exp_value)),
    )
