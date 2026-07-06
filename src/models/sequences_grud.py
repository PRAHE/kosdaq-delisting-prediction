"""
S1 Phase 2 — GRU-D용 시퀀스 빌더 (Che et al. 2018).

Phase 1과 다른 점:
  - combined_raw.csv에서 시작해 **결측을 NaN 그대로 보존**
  - 거시변수 6개는 fixed_v1에서 join (NaN 없음)
  - 각 (B, K, D)에 대해 feature-level mask M_{t,d} ∈ {0,1}와
    time gap δ_{t,d}를 함께 산출

  δ_{t,d} = feature d가 t시점에 관측됐는지 여부 + 이전 관측까지의 timestep 거리.
    관측 → δ = 0
    결측 → δ = (t - last_observed_t for feature d) in timesteps

  X̂는 NaN 자리를 0으로 채움 (모델에서 decay로 사용)
  X_last (B, K, D) = 각 (t, d)에서의 "가장 최근에 관측된 값"
  X_mean (D,) = train의 feature별 평균 (decay target)

라벨링 / 평가 분할은 Phase 1과 동일하게 fixed_v1 인덱스 사용.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.s0_diagnostic.baseline import signed_log1p
from src.features.labelings import (
    LABELING_FUNCTIONS,
    load_panel_with_delta,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
COMBINED_PATH = PROJECT_ROOT / "preprocess" / "data" / "processed" / "combined_raw.csv"

META = {"stock_code", "year", "quarter", "gics_sector", "label"}
HIGH_MISSING = {"매출액증가율", "순이익증가율", "영업이익증가율"}
MACRO_COLS = ["credit_spread", "kosdaq_return", "gdp_growth_yoy",
              "usdkrw_chg", "vix_avg", "cpi_yoy"]

DEFAULT_K = 3


@dataclass
class GRUDSequenceDataset:
    """(B, K, D) 시퀀스 + 결측 처리 부속 행렬."""
    X: np.ndarray          # (N, K, D)  결측 자리 = 0
    M: np.ndarray          # (N, K, D)  관측 여부 (1/0)
    X_last: np.ndarray     # (N, K, D)  각 (t, d)에서의 가장 최근 관측값 (없으면 0)
    Delta: np.ndarray      # (N, K, D)  결측 timestep 거리 (관측 = 0)
    y: np.ndarray          # (N,)
    keys: pd.DataFrame
    feature_cols: list[str]
    X_mean: np.ndarray     # (D,)       train의 feature mean (decay target)


def _load_combined_with_macro() -> pd.DataFrame:
    """combined_raw에서 NaN 보존 + 거시변수를 fixed_v1에서 join하여 합친 panel."""
    df = pd.read_csv(COMBINED_PATH, dtype={"stock_code": str})

    # 거시변수는 fixed_v1의 다양한 split에 동일 (시점만으로 결정)
    fixed_parts = []
    for split in ["train", "valid", "test"]:
        fp = PROJECT_ROOT / "preprocess" / "data" / "processed_fixed_v1" / "fixed_N1" / "exp-A" / f"{split}.csv"
        sub = pd.read_csv(fp, dtype={"stock_code": str})
        fixed_parts.append(sub[["stock_code", "year", "quarter"] + MACRO_COLS])
    macro = pd.concat(fixed_parts, ignore_index=True).drop_duplicates(
        subset=["stock_code", "year", "quarter"]
    )
    df = df.merge(macro, on=["stock_code", "year", "quarter"], how="left")
    return df


def _build_index(panel: pd.DataFrame) -> dict:
    """(stock_code, quarter) → {year: row_idx}"""
    idx: dict[tuple[str, str], dict[int, int]] = {}
    for i, r in enumerate(panel[["stock_code", "year", "quarter"]].itertuples(index=False)):
        idx.setdefault((r.stock_code, r.quarter), {})[int(r.year)] = i
    return idx


def _select_features(panel: pd.DataFrame) -> list[str]:
    return [c for c in panel.columns
            if c not in META and c not in HIGH_MISSING
            and c not in ("delist_year", "delta")]


def build_grud_sequences(
    target_panel: pd.DataFrame,
    history_panel: pd.DataFrame,
    feature_cols: list[str],
    label_fn,
    X_mean: np.ndarray,
    K: int = DEFAULT_K,
    apply_log1p: bool = True,
) -> GRUDSequenceDataset:
    """target_panel의 각 행에 대해 K timestep 시퀀스 + (X, M, X_last, Delta) 구성.

    history_panel은 NaN을 보존한 raw values를 가진다.
    apply_log1p=True면 관측된 값에 signed_log1p 적용 (Phase 1과 일관 scaling).
    """
    F = len(feature_cols)
    hist_X = history_panel[feature_cols].values.astype(np.float32)  # NaN 보존
    hist_M = (~np.isnan(hist_X)).astype(np.float32)
    # NaN 자리를 0으로 채움 (mask가 0이라 모델에서 무시되지만 안전상 0)
    hist_X = np.nan_to_num(hist_X, nan=0.0)
    if apply_log1p:
        hist_X = signed_log1p(hist_X).astype(np.float32)

    hist_idx = _build_index(history_panel)

    N = len(target_panel)
    X = np.zeros((N, K, F), dtype=np.float32)
    M = np.zeros((N, K, F), dtype=np.float32)
    X_last = np.zeros((N, K, F), dtype=np.float32)
    Delta = np.zeros((N, K, F), dtype=np.float32)

    for n, r in enumerate(target_panel[["stock_code", "year", "quarter"]].itertuples(index=False)):
        stock = r.stock_code
        cur_year = int(r.year)
        quarter = r.quarter
        idx_map = hist_idx.get((stock, quarter), {})

        # feature별로 last_observed value와 last_observed timestep 기록
        last_val = np.copy(X_mean)  # 초기엔 mean으로
        last_t = np.full(F, -1, dtype=np.int32)  # 한 번도 관측 X

        for k in range(K):
            yr = cur_year - (K - 1 - k)
            if yr in idx_map:
                ridx = idx_map[yr]
                x_raw = hist_X[ridx]    # (F,)
                m_raw = hist_M[ridx]    # (F,)
                # X, M
                X[n, k] = x_raw  # raw value (NaN은 이미 0)
                M[n, k] = m_raw
                # X_last (각 feature별로)
                X_last[n, k] = np.where(m_raw > 0, x_raw, last_val)
                # Delta = feature별 결측 거리
                # 관측이면 0, 결측이면 (k - last_t) 단, last_t < 0이면 (k + 1)
                d = np.where(
                    m_raw > 0,
                    0.0,
                    np.where(last_t < 0, float(k + 1), (k - last_t).astype(np.float32)),
                )
                Delta[n, k] = d
                # update last_val, last_t
                obs_mask = m_raw > 0
                last_val = np.where(obs_mask, x_raw, last_val)
                last_t = np.where(obs_mask, k, last_t)
            else:
                # timestep 자체가 없으면 mask = 0, X = 0, X_last = 이전 last_val
                X[n, k] = 0
                M[n, k] = 0
                X_last[n, k] = last_val
                Delta[n, k] = np.where(last_t < 0, float(k + 1), (k - last_t).astype(np.float32))
                # last 미갱신

    y = label_fn(target_panel)
    return GRUDSequenceDataset(
        X=X, M=M, X_last=X_last, Delta=Delta,
        y=np.asarray(y),
        keys=target_panel[["stock_code", "year", "quarter"]].reset_index(drop=True),
        feature_cols=feature_cols,
        X_mean=X_mean.astype(np.float32),
    )


def prepare_phase2_datasets(
    labeling: str = "L3_rolling_H24",
    K: int = DEFAULT_K,
) -> dict:
    """Phase 2 train/valid/test GRU-D 시퀀스 데이터셋.

    train panel: fixed_v1 train (라벨이 정의된 (stock,year,quarter)).
                 단 2025 행 제외 (forward-looking 라벨링).
    history panel: combined_raw + 거시변수, NaN 보존.
    """
    train_panel = load_panel_with_delta("train")
    valid_panel = load_panel_with_delta("valid")
    test_panel  = load_panel_with_delta("test")
    train_panel = train_panel[train_panel["year"].isin(range(2015, 2024))].reset_index(drop=True)

    combined = _load_combined_with_macro()
    feature_cols = _select_features(combined)

    # X_mean: train에서 (stock,year,quarter) 키만 combined에서 가져와 feature 평균 계산
    # signed_log1p 적용 후 값 기준으로 mean 계산 (모델 입력 scale과 일치)
    train_keys = train_panel[["stock_code", "year", "quarter"]]
    train_combined = combined.merge(train_keys, on=["stock_code", "year", "quarter"], how="inner")
    raw = train_combined[feature_cols].values.astype(np.float32)
    mask = ~np.isnan(raw)
    # signed_log1p 적용 (NaN 위치는 그대로 NaN, mean에서 skip)
    raw_log = np.where(mask, signed_log1p(np.nan_to_num(raw, nan=0.0)), np.nan)
    X_mean = np.nanmean(raw_log, axis=0)
    X_mean = np.nan_to_num(X_mean, nan=0.0)

    train_label_fn = LABELING_FUNCTIONS[labeling][0]
    eval_label_fn = LABELING_FUNCTIONS["L1_fixed_N1"][0]

    return {
        "train": build_grud_sequences(train_panel, combined, feature_cols,
                                       train_label_fn, X_mean, K=K),
        "valid": build_grud_sequences(valid_panel, combined, feature_cols,
                                       eval_label_fn, X_mean, K=K),
        "test":  build_grud_sequences(test_panel, combined, feature_cols,
                                       eval_label_fn, X_mean, K=K),
        "feature_cols": feature_cols,
        "X_mean": X_mean,
        "labeling": labeling,
    }


def summarize(ds: GRUDSequenceDataset, name: str = "") -> str:
    overall_obs_ratio = float(ds.M.mean())
    feat_avg_obs = ds.M.mean(axis=(0, 1))
    most_missing = np.argsort(feat_avg_obs)[:3]
    least_missing = np.argsort(feat_avg_obs)[::-1][:3]
    fc = ds.feature_cols
    return (
        f"{name}: N={len(ds.y)}  K={ds.X.shape[1]}  D={ds.X.shape[2]}  "
        f"pos={int((ds.y > 0).sum())}  "
        f"obs_ratio={overall_obs_ratio:.3f}\n"
        f"  most missing: {[(fc[i], f'{feat_avg_obs[i]:.2f}') for i in most_missing]}\n"
        f"  most observed: {[(fc[i], f'{feat_avg_obs[i]:.2f}') for i in least_missing]}"
    )


if __name__ == "__main__":
    print("[load] Phase 2 GRU-D datasets")
    data = prepare_phase2_datasets()
    print(summarize(data["train"], "train"))
    print(summarize(data["valid"], "valid"))
    print(summarize(data["test"],  "test "))
