"""
Stream R — 라벨링 변환 함수.

모든 라벨링을 fixed_v1 train 패널(35,493행) 위에서 derive한다.
같은 (stock_code, year, quarter) 키를 N1/N2/N3가 공유하므로,
세 데이터의 label을 join하면 각 행에 대한 `delist_year - year` 거리를 알 수 있다.

7개 스킴:
  L1: fixed_N1 (현재 baseline)         — label = 1 if Δ==1
  L2: rolling_H12 (~1년 내)            — label = 1 if Δ ∈ {0, 1}
  L3: rolling_H24 (~2년 내)            — label = 1 if Δ ∈ {0, 1, 2}
  L4: fixed_N1 ∪ N2 ∪ N3               — label = 1 if Δ ∈ {1, 2, 3}
  L5: ordinal 4-class                  — label ∈ {0, 1, 2, 3} (3 = N1 가장 위험)
  L6: continuous risk exp(-Δ/τ)        — label ∈ [0, 1], τ=2
  L7: v3-style all years of delisted   — label = 1 if firm is ever delisted

Δ = delist_year - year (양수면 미래 상폐, 0이면 해당 연도 상폐, NaN이면 정상)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIXED_DATA_DIR = PROJECT_ROOT / "preprocess" / "data" / "processed_fixed_v1"

KEY_COLS = ["stock_code", "year", "quarter"]


def load_panel_with_delta(split: str = "train", variant: str = "exp-A") -> pd.DataFrame:
    """(stock_code, year, quarter) 패널에 `delist_year`, `delta` 컬럼을 부착해 반환.

    N1/N2/N3 데이터의 label 컬럼을 join하여 각 행이 어떤 N-year 라벨을 받았는지
    파악하고, 이를 통해 delist_year를 역산한다.

    delta = delist_year - year  (정상 기업은 NaN)
    """
    n1 = pd.read_csv(FIXED_DATA_DIR / "fixed_N1" / variant / f"{split}.csv",
                     dtype={"stock_code": str})
    n2 = pd.read_csv(FIXED_DATA_DIR / "fixed_N2" / variant / f"{split}.csv",
                     dtype={"stock_code": str})
    n3 = pd.read_csv(FIXED_DATA_DIR / "fixed_N3" / variant / f"{split}.csv",
                     dtype={"stock_code": str})

    # 양성 행에서 delta 도출: label=1 in Nk → delta == k
    def _stock_delist_year_from(df: pd.DataFrame, k: int) -> dict[str, int]:
        # df에서 label=1인 행의 year + k = delist_year
        pos = df[df["label"] == 1][["stock_code", "year"]].copy()
        pos["delist_year"] = pos["year"] + k
        # stock_code별 최소 delist_year (가장 이른 상폐) — 동일 회사 여러 행 있을 수 있음
        return pos.groupby("stock_code")["delist_year"].min().to_dict()

    delist_map: dict[str, int] = {}
    for df, k in [(n1, 1), (n2, 2), (n3, 3)]:
        for stock, dy in _stock_delist_year_from(df, k).items():
            # 가장 정확한 정보 우선: N1이 가장 정확 (1년 전 = delist_year-1).
            # 따라서 N1, N2, N3 순으로 update — N1이 마지막에 덮어쓰지 못하게 처음 등록 우선.
            if stock not in delist_map:
                delist_map[stock] = dy
            else:
                # 서로 모순이면 더 작은(이른) delist_year 채택
                delist_map[stock] = min(delist_map[stock], dy)

    panel = n1.copy()
    panel["delist_year"] = panel["stock_code"].map(delist_map)
    panel["delta"] = panel["delist_year"] - panel["year"]
    return panel


# ---------------------------------------------------------------------------
# 라벨링 변환 함수: 입력은 load_panel_with_delta로 만든 panel, 출력은 새 label 컬럼
# ---------------------------------------------------------------------------


def label_L1_fixed_N1(panel: pd.DataFrame) -> np.ndarray:
    return (panel["delta"] == 1).astype(int).values


def label_L2_rolling_H12(panel: pd.DataFrame) -> np.ndarray:
    """≤1년 내 상폐: delta ∈ {0, 1}. 우리 패널에서 delta=0 행은 없을 수 있으나
    있다면 같이 양성으로 표시 (상폐 당해년 데이터)."""
    return panel["delta"].isin([0, 1]).astype(int).values


def label_L3_rolling_H24(panel: pd.DataFrame) -> np.ndarray:
    return panel["delta"].isin([0, 1, 2]).astype(int).values


def label_L4_union_N1_N2_N3(panel: pd.DataFrame) -> np.ndarray:
    return panel["delta"].isin([1, 2, 3]).astype(int).values


def label_L5_ordinal(panel: pd.DataFrame) -> np.ndarray:
    """순서형 4-class:
      delta == 1 → 3 (가장 위험)
      delta == 2 → 2
      delta == 3 → 1
      그 외     → 0
    """
    mapping = {1: 3, 2: 2, 3: 1}
    return panel["delta"].map(mapping).fillna(0).astype(int).values


def label_L6_continuous_risk(panel: pd.DataFrame, tau: float = 2.0) -> np.ndarray:
    """연속 위험도: delta ∈ {1,2,3}이면 exp(-delta/τ), 그 외 0.
    τ=2 → delta=1: 0.607, delta=2: 0.368, delta=3: 0.223
    """
    raw = np.zeros(len(panel), dtype=float)
    mask = panel["delta"].isin([1, 2, 3])
    raw[mask.values] = np.exp(-panel.loc[mask, "delta"].values / tau)
    return raw


def label_L7_v3_all_years(panel: pd.DataFrame) -> np.ndarray:
    """v3 스타일: 상폐된 기업의 모든 train year 행 = 1. 단 year ≤ delist_year만 (미래 누수 차단)."""
    has_delist = panel["delist_year"].notna()
    not_after_delist = panel["year"] <= panel["delist_year"].fillna(panel["year"].max() + 100)
    return (has_delist & not_after_delist).astype(int).values


LABELING_FUNCTIONS = {
    "L1_fixed_N1":         (label_L1_fixed_N1,         "binary"),
    "L2_rolling_H12":      (label_L2_rolling_H12,      "binary"),
    "L3_rolling_H24":      (label_L3_rolling_H24,      "binary"),
    "L4_union_N1N2N3":     (label_L4_union_N1_N2_N3,   "binary"),
    "L5_ordinal":          (label_L5_ordinal,          "ordinal"),
    "L6_continuous":       (label_L6_continuous_risk,  "continuous"),
    "L7_v3_all_years":     (label_L7_v3_all_years,     "binary"),
}


def summarize_labelings(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, (fn, ttype) in LABELING_FUNCTIONS.items():
        y = fn(panel)
        if ttype == "continuous":
            n_pos_like = int((y > 0).sum())
            ratio = round(float((y > 0).mean()), 4)
            mean_score = round(float(y.mean()), 4)
            rows.append({"labeling": name, "type": ttype, "n_nonzero": n_pos_like,
                         "nonzero_ratio": ratio, "mean_score": mean_score})
        elif ttype == "ordinal":
            counts = pd.Series(y).value_counts().sort_index()
            rows.append({"labeling": name, "type": ttype, "n_total": len(y),
                         **{f"class_{k}": int(counts.get(k, 0)) for k in [0,1,2,3]}})
        else:
            n_pos = int((y == 1).sum())
            rows.append({"labeling": name, "type": ttype, "n_total": len(y),
                         "n_pos": n_pos, "pos_ratio": round(n_pos / len(y), 4),
                         "imbalance_ratio": round((len(y) - n_pos) / max(n_pos, 1), 2)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    for split in ["train", "valid", "test"]:
        print(f"\n=== {split} panel ===")
        panel = load_panel_with_delta(split)
        print(f"  rows={len(panel)}, n_delisted_firms={panel['delist_year'].notna().sum() and panel['stock_code'][panel['delist_year'].notna()].nunique()}")
        print(summarize_labelings(panel).to_string(index=False))
