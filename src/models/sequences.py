"""
S1 Phase 1 — 시퀀스 빌더.

각 (stock_code, year, quarter) 샘플에 대해 **동일 quarter의 과거 K년 시계열**을
구성한다 (current year 포함). 예: (008800, 2020, Q1), K=3 →
[(2018 Q1), (2019 Q1), (2020 Q1)] (마지막이 current).

데이터가 부족한 경우 zero-pad + mask. K개 미만이면 앞쪽이 mask=0.

기존 fixed_N1 train/valid/test의 모든 행을 보존하며, sequence/mask/label만 추가.
스케일링: 기존 S0와 동일하게 SimpleImputer(median) + signed_log1p, train에서 fit.

라벨링은 L1 fixed_N1 (default) 또는 L3 rolling_H24 등 sr_labeling의 함수 호출.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from src.research.s0_diagnostic.baseline import (
    HIGH_MISSING_COLS,
    META_COLUMNS,
    TARGET_COLUMN,
    signed_log1p,
)
from src.features.labelings import (
    LABELING_FUNCTIONS,
    load_panel_with_delta,
)

MARKET_FEATURES_PATH = (
    Path(__file__).resolve().parents[3]
    / "preprocess" / "data" / "market" / "market_features.csv"
)
MARKET_FEATURE_COLS = [
    "price_log_close",
    "price_ret_12m",
    "price_volatility_60d",
    "price_drawdown_max_12m",
    "volume_log_mean_60d",
    "volume_change_yoy",
]

AUDIT_FEATURES_PATH = (
    Path(__file__).resolve().parents[3]
    / "preprocess" / "data" / "audit" / "audit_features.csv"
)
AUDIT_FEATURE_COLS = [
    "audit_opinion_t",
    "audit_opinion_t1",
    "audit_nonclean_consec",
    "audit_nonclean_5y",
    "audit_observed",
]

SUPERVISION_FEATURES_PATH = (
    Path(__file__).resolve().parents[3]
    / "preprocess" / "data" / "supervision" / "supervision_features.csv"
)
SUPERVISION_FEATURE_COLS = [
    "is_supervised_now",
    "days_since_last_concern",
    "n_supervision_events_5y",
    "n_concern_events_5y",
    "has_trading_halt_3y",
    "has_any_supervision_history",
]


def _attach_market(panel: pd.DataFrame, with_market: bool) -> pd.DataFrame:
    """panel에 market_features.csv를 left-join하여 6개 시장 피처 추가."""
    if not with_market:
        return panel
    if not MARKET_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"with_market=True인데 {MARKET_FEATURES_PATH}가 없음. "
            "`python -m src.research.a3_market.market_features`로 먼저 생성하세요."
        )
    mkt = pd.read_csv(MARKET_FEATURES_PATH, dtype={"stock_code": str})
    mkt["stock_code"] = mkt["stock_code"].str.zfill(6)
    panel = panel.copy()
    panel["stock_code"] = panel["stock_code"].astype(str).str.zfill(6)
    return panel.merge(mkt, on=["stock_code", "year", "quarter"], how="left")


def _attach_audit(panel: pd.DataFrame, with_audit: bool) -> pd.DataFrame:
    """panel에 audit_features.csv를 left-join하여 5개 감사 피처 추가."""
    if not with_audit:
        return panel
    if not AUDIT_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"with_audit=True인데 {AUDIT_FEATURES_PATH}가 없음. "
            "`python -m src.research.a1_audit.audit_features`로 먼저 생성하세요."
        )
    aud = pd.read_csv(AUDIT_FEATURES_PATH, dtype={"stock_code": str})
    aud["stock_code"] = aud["stock_code"].str.zfill(6)
    panel = panel.copy()
    panel["stock_code"] = panel["stock_code"].astype(str).str.zfill(6)
    return panel.merge(aud, on=["stock_code", "year", "quarter"], how="left")


def _attach_supervision(panel: pd.DataFrame, with_supervision: bool) -> pd.DataFrame:
    """panel에 supervision_features.csv를 left-join하여 6개 관리종목 피처 추가."""
    if not with_supervision:
        return panel
    if not SUPERVISION_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"with_supervision=True인데 {SUPERVISION_FEATURES_PATH}가 없음. "
            "`python -m src.research.a2_supervision.supervision_features`로 먼저 생성하세요."
        )
    sup = pd.read_csv(SUPERVISION_FEATURES_PATH, dtype={"stock_code": str})
    sup["stock_code"] = sup["stock_code"].str.zfill(6)
    panel = panel.copy()
    panel["stock_code"] = panel["stock_code"].astype(str).str.zfill(6)
    return panel.merge(sup, on=["stock_code", "year", "quarter"], how="left")

DEFAULT_K = 3
EXCLUDE_COLS = META_COLUMNS | {TARGET_COLUMN, "delist_year", "delta"} | HIGH_MISSING_COLS


@dataclass
class SequenceDataset:
    """K-step 시계열 데이터셋.

    X: (N, K, F)  — float32, signed_log1p 적용된 피처
    M: (N, K)     — int8, mask (1 = 관측, 0 = padding)
    y: (N,)       — int 또는 float (binary/ordinal/continuous)
    keys: (N, 3)  — (stock_code, year, quarter)
    """
    X: np.ndarray
    M: np.ndarray
    y: np.ndarray
    keys: pd.DataFrame
    feature_cols: list[str]
    K: int


def _select_features(panel: pd.DataFrame) -> list[str]:
    return [c for c in panel.columns if c not in EXCLUDE_COLS]


def _build_history_index(panel: pd.DataFrame) -> dict:
    """(stock_code, quarter) → {year: row_index}"""
    idx: dict[tuple[str, str], dict[int, int]] = {}
    for i, row in enumerate(panel[["stock_code", "year", "quarter"]].itertuples(index=False)):
        key = (row.stock_code, row.quarter)
        idx.setdefault(key, {})[int(row.year)] = i
    return idx


def build_sequences(
    target_panel: pd.DataFrame,
    history_panel: pd.DataFrame,
    feature_cols: list[str],
    imputer: SimpleImputer,
    label_fn,
    K: int = DEFAULT_K,
    apply_log1p: bool = True,
) -> SequenceDataset:
    """target_panel의 각 행에 대해, history_panel에서 동일 (stock_code, quarter)의
    과거 K년 시퀀스를 구성한다.

    Args:
        target_panel: 라벨이 부여될 패널 (예: fixed_N1 train의 행들)
        history_panel: 시퀀스를 만들 때 참조할 패널 (예: combined_raw 또는 같은 train)
        imputer: 이미 train에서 fit된 SimpleImputer
        label_fn: target_panel에 대해 label을 반환하는 함수 (e.g. label_L1_fixed_N1)

    Returns:
        SequenceDataset
    """
    # history_panel의 raw features를 impute + log1p
    X_hist_imputed = imputer.transform(history_panel[feature_cols])
    if apply_log1p:
        X_hist_imputed = signed_log1p(X_hist_imputed)
    X_hist_imputed = X_hist_imputed.astype(np.float32)

    history_idx = _build_history_index(history_panel)

    N = len(target_panel)
    F = len(feature_cols)
    X = np.zeros((N, K, F), dtype=np.float32)
    M = np.zeros((N, K), dtype=np.int8)

    for n, row in enumerate(target_panel[["stock_code", "year", "quarter"]].itertuples(index=False)):
        stock = row.stock_code
        cur_year = int(row.year)
        quarter = row.quarter
        idx_for_pair = history_idx.get((stock, quarter), {})
        # K개 timestep: cur_year - (K-1) ... cur_year
        for k in range(K):
            yr = cur_year - (K - 1 - k)  # k=0이면 가장 과거, k=K-1이면 current
            if yr in idx_for_pair:
                src_idx = idx_for_pair[yr]
                X[n, k, :] = X_hist_imputed[src_idx]
                M[n, k] = 1
            # else: zero + mask=0

    y = label_fn(target_panel)

    keys = target_panel[["stock_code", "year", "quarter"]].reset_index(drop=True)
    return SequenceDataset(X=X, M=M, y=np.asarray(y), keys=keys,
                            feature_cols=feature_cols, K=K)


def prepare_fold_datasets(
    fold_id: int,
    train_year_range: tuple[int, int],
    valid_year: int,
    labeling: str = "L3_rolling_H24",
    K: int = DEFAULT_K,
    with_market: bool = False,
    with_audit: bool = False,
    with_supervision: bool = False,
) -> dict:
    """Walk-forward fold: train year cutoff + valid year.

    - train_panel = full train의 year ∈ [yr_lo, yr_hi]
    - valid_panel = full train 또는 full valid에서 year == valid_year인 행
    - test_panel = full test (2024)
    - imputer는 fold별 train에서 fit
    - history panel = train+valid+test의 합 (sequence가 과거만 참조하므로 누수 없음)
    """
    full_train = load_panel_with_delta("train")
    full_train = full_train[full_train["year"].isin(range(2015, 2024))].reset_index(drop=True)
    full_valid = load_panel_with_delta("valid")
    full_test  = load_panel_with_delta("test")

    # 시장 피처 join
    full_train = _attach_market(full_train, with_market)
    full_valid = _attach_market(full_valid, with_market)
    full_test  = _attach_market(full_test,  with_market)
    # 감사 피처 join
    full_train = _attach_audit(full_train, with_audit)
    full_valid = _attach_audit(full_valid, with_audit)
    full_test  = _attach_audit(full_test,  with_audit)
    # 관리종목 피처 join
    full_train = _attach_supervision(full_train, with_supervision)
    full_valid = _attach_supervision(full_valid, with_supervision)
    full_test  = _attach_supervision(full_test,  with_supervision)

    yr_lo, yr_hi = train_year_range
    train_panel = full_train[
        (full_train["year"] >= yr_lo) & (full_train["year"] <= yr_hi)
    ].reset_index(drop=True)

    if valid_year in full_train["year"].unique():
        valid_panel = full_train[full_train["year"] == valid_year].reset_index(drop=True)
    else:
        valid_panel = full_valid[full_valid["year"] == valid_year].reset_index(drop=True)

    test_panel = full_test  # 2024 전체

    feature_cols = _select_features(train_panel)
    imputer = SimpleImputer(strategy="median")
    imputer.fit(train_panel[feature_cols])

    history_panel = pd.concat(
        [full_train, full_valid, full_test], ignore_index=True
    ).drop_duplicates(subset=["stock_code", "year", "quarter"]).reset_index(drop=True)

    train_label_fn = LABELING_FUNCTIONS[labeling][0]
    eval_label_fn, _ = LABELING_FUNCTIONS["L1_fixed_N1"]

    return {
        "fold_id": fold_id,
        "train_year_range": train_year_range,
        "valid_year": valid_year,
        "train": build_sequences(train_panel, history_panel, feature_cols, imputer,
                                  train_label_fn, K=K),
        "valid": build_sequences(valid_panel, history_panel, feature_cols, imputer,
                                  eval_label_fn, K=K),
        "test": build_sequences(test_panel, history_panel, feature_cols, imputer,
                                 eval_label_fn, K=K),
        "feature_cols": feature_cols,
        "imputer": imputer,
        "labeling": labeling,
    }


def prepare_phase1_datasets(
    labeling: str = "L3_rolling_H24",
    K: int = DEFAULT_K,
    with_market: bool = False,
    with_audit: bool = False,
    with_supervision: bool = False,
) -> dict[str, SequenceDataset]:
    """Phase 1 학습용 train/valid/test 시퀀스 데이터셋을 모두 준비한다.

    - train panel: fixed_N1 train + 추가로 2025 행은 제외 (S0/SR와 동일)
    - history (시퀀스 ref): train+valid+test 패널 결합 → 어떤 시점의 K년 과거도 찾을 수 있게
      (단 imputer/scale은 train에서만 fit)
    - 라벨: labeling으로 train에 부여, valid/test는 항상 L1_fixed_N1
    """
    train_panel = load_panel_with_delta("train")
    valid_panel = load_panel_with_delta("valid")
    test_panel  = load_panel_with_delta("test")

    # train에서 2025 행 제외 (forward-looking 라벨)
    train_panel = train_panel[train_panel["year"].isin(range(2015, 2024))].reset_index(drop=True)

    # 시장 피처 join
    train_panel = _attach_market(train_panel, with_market)
    valid_panel = _attach_market(valid_panel, with_market)
    test_panel  = _attach_market(test_panel,  with_market)
    # 감사 피처 join
    train_panel = _attach_audit(train_panel, with_audit)
    valid_panel = _attach_audit(valid_panel, with_audit)
    test_panel  = _attach_audit(test_panel,  with_audit)
    # 관리종목 피처 join
    train_panel = _attach_supervision(train_panel, with_supervision)
    valid_panel = _attach_supervision(valid_panel, with_supervision)
    test_panel  = _attach_supervision(test_panel,  with_supervision)

    feature_cols = _select_features(train_panel)

    # imputer는 train의 feature columns에서만 fit
    imputer = SimpleImputer(strategy="median")
    imputer.fit(train_panel[feature_cols])

    # history panel: train + valid + test 결합 — 각 행의 과거 K년 sequence를 빌드할 때
    # 보다 풍부한 history 확보 (단 시간 누수는 없다: 과거 행만 인덱싱하므로)
    history_panel = pd.concat(
        [train_panel, valid_panel, test_panel],
        ignore_index=True,
    ).drop_duplicates(subset=["stock_code", "year", "quarter"]).reset_index(drop=True)

    train_label_fn, _ = LABELING_FUNCTIONS[labeling]
    eval_label_fn, _ = LABELING_FUNCTIONS["L1_fixed_N1"]

    return {
        "train": build_sequences(train_panel, history_panel, feature_cols, imputer,
                                  train_label_fn, K=K),
        "valid": build_sequences(valid_panel, history_panel, feature_cols, imputer,
                                  eval_label_fn, K=K),
        "test": build_sequences(test_panel, history_panel, feature_cols, imputer,
                                 eval_label_fn, K=K),
        "feature_cols": feature_cols,
        "imputer": imputer,
        "labeling": labeling,
    }


def summarize(ds: SequenceDataset, name: str = "") -> str:
    nz_steps = ds.M.sum(axis=1)
    return (
        f"{name}: N={len(ds.y)}  K={ds.K}  F={len(ds.feature_cols)}  "
        f"pos={int((ds.y > 0).sum())}  "
        f"mean_obs_steps={nz_steps.mean():.2f}  full_seq={(nz_steps == ds.K).sum()}"
    )


if __name__ == "__main__":
    print("[load] Phase 1 datasets with labeling=L3_rolling_H24, K=3")
    data = prepare_phase1_datasets()
    print(summarize(data["train"], "train"))
    print(summarize(data["valid"], "valid"))
    print(summarize(data["test"],  "test "))
