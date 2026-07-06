"""
PR-AUC의 신뢰구간 추정 — Bayesian bootstrap + Frequentist bootstrap.

Bayesian bootstrap (Rubin 1981):
    각 resample에서 sample weight를 Dirichlet(1,...,1) 분포로 sampling.
    이산 frequentist bootstrap과 달리 가중치가 연속적이고
    sample size가 작을 때 ranking-metric 분포가 더 안정적이다.

평가 지표:
    - PR-AUC: sklearn.metrics.average_precision_score (가중 버전 사용)
    - ROC-AUC: sklearn.metrics.roc_auc_score (가중 버전 사용)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass
class BootstrapResult:
    metric: str
    method: str
    point_estimate: float
    mean: float
    median: float
    ci95_lo: float
    ci95_hi: float
    std: float
    n_boot: int
    posterior: np.ndarray


def _ap_weighted(y_true: np.ndarray, y_proba: np.ndarray, w: np.ndarray) -> float:
    return float(average_precision_score(y_true, y_proba, sample_weight=w))


def _roc_weighted(y_true: np.ndarray, y_proba: np.ndarray, w: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_proba, sample_weight=w))


_METRIC_FN = {"pr_auc": _ap_weighted, "roc_auc": _roc_weighted}


def bayesian_bootstrap(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = "pr_auc",
    n_boot: int = 1000,
    seed: int = 42,
) -> BootstrapResult:
    """Dirichlet(1,...,1) sample weight 기반 Bayesian bootstrap.

    각 resample에서 n개 sample weight w ~ Dirichlet(1,...,1)을 뽑아
    metric을 weighted 형태로 계산한다.
    """
    if metric not in _METRIC_FN:
        raise ValueError(f"unknown metric: {metric}")
    metric_fn = _METRIC_FN[metric]

    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    n = len(y_true)

    rng = np.random.default_rng(seed)
    posterior = np.empty(n_boot, dtype=np.float64)

    # Dirichlet(1,...,1)을 vectorized로: Gamma(1,1) -> 정규화
    gammas = rng.standard_gamma(shape=1.0, size=(n_boot, n))
    weights = gammas / gammas.sum(axis=1, keepdims=True)
    weights *= n  # sum to n, so weighted metric numerically matches scale of unweighted

    for i in range(n_boot):
        posterior[i] = metric_fn(y_true, y_proba, weights[i])

    point = metric_fn(y_true, y_proba, np.ones(n))
    return BootstrapResult(
        metric=metric,
        method="bayesian_dirichlet",
        point_estimate=point,
        mean=float(np.mean(posterior)),
        median=float(np.median(posterior)),
        ci95_lo=float(np.percentile(posterior, 2.5)),
        ci95_hi=float(np.percentile(posterior, 97.5)),
        std=float(np.std(posterior)),
        n_boot=n_boot,
        posterior=posterior,
    )


def frequentist_bootstrap(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = "pr_auc",
    n_boot: int = 1000,
    seed: int = 42,
    stratified: bool = True,
) -> BootstrapResult:
    """전통적 with-replacement bootstrap.

    stratified=True인 경우 positive/negative 비율을 보존하여 sample.
    극심한 불균형 환경에서 빈 양성 sample을 방지한다.
    """
    if metric not in _METRIC_FN:
        raise ValueError(f"unknown metric: {metric}")
    metric_fn = _METRIC_FN[metric]

    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    n = len(y_true)
    rng = np.random.default_rng(seed)

    if stratified:
        pos_idx = np.where(y_true == 1)[0]
        neg_idx = np.where(y_true == 0)[0]
        n_pos, n_neg = len(pos_idx), len(neg_idx)
    posterior = np.empty(n_boot, dtype=np.float64)

    for i in range(n_boot):
        if stratified:
            idx = np.concatenate(
                [
                    rng.choice(pos_idx, size=n_pos, replace=True),
                    rng.choice(neg_idx, size=n_neg, replace=True),
                ]
            )
        else:
            idx = rng.integers(0, n, size=n)
        w = np.ones(len(idx))
        posterior[i] = metric_fn(y_true[idx], y_proba[idx], w)

    point = metric_fn(y_true, y_proba, np.ones(n))
    return BootstrapResult(
        metric=metric,
        method="frequentist_stratified" if stratified else "frequentist",
        point_estimate=point,
        mean=float(np.mean(posterior)),
        median=float(np.median(posterior)),
        ci95_lo=float(np.percentile(posterior, 2.5)),
        ci95_hi=float(np.percentile(posterior, 97.5)),
        std=float(np.std(posterior)),
        n_boot=n_boot,
        posterior=posterior,
    )


def to_dict(result: BootstrapResult, include_posterior: bool = False) -> dict:
    d = {
        "metric": result.metric,
        "method": result.method,
        "point_estimate": result.point_estimate,
        "mean": result.mean,
        "median": result.median,
        "ci95_lo": result.ci95_lo,
        "ci95_hi": result.ci95_hi,
        "std": result.std,
        "n_boot": result.n_boot,
    }
    if include_posterior:
        d["posterior"] = result.posterior.tolist()
    return d
