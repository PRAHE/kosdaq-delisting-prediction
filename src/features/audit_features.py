"""
A1 외부 데이터 — 분기별 감사의견 피처 집계.

캐시된 (corp_code, year) 감사의견 JSON을 읽어 (stock_code, year, quarter)
단위 피처 5개로 변환한다.

**시점 정합성 (look-ahead bias 차단)**:
사업보고서(감사보고서 포함)는 회계연도 다음 해 3월 말에 공개된다.
따라서 (stock, year=Y, quarter=*)의 시점에서 사용 가능한 감사의견은:
  - year=Y의 모든 quarter (Q1=3월말, H1=6월, Q3=9월, ANNUAL=12월)
    에서 **가장 최근 사용 가능한 감사의견 = 회계연도 Y-1 의견** (Y-1년 사업보고서)
  - 그 직전 = 회계연도 Y-2 의견
  - …

피처 5개:
  audit_opinion_t        — 직전(Y-1) 감사의견 정수 (0=적정, 1=한정, 2=부적정, 3=의견거절)
  audit_opinion_t1       — 그 이전(Y-2) 감사의견 정수
  audit_nonclean_consec  — Y-1에서 거꾸로 거슬러 가며 비적정이 연속된 횟수 (0~5)
  audit_nonclean_5y      — Y-5 ~ Y-1 5년간 비적정 의견 횟수 (0~5)
  audit_observed         — Y-1 감사의견 데이터 존재 여부 (1/0)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.a1_audit.corp_map import build_stock_to_corp, load_panel_stock_codes

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = PROJECT_ROOT / "data" / "audit_opinion"
OUT_DIR = PROJECT_ROOT / "preprocess" / "data" / "audit"

OPINION_TO_INT = {
    "적정": 0,
    "한정": 1,
    "부적정": 2,
    "의견거절": 3,
}

AUDIT_FEATURE_COLS = [
    "audit_opinion_t",
    "audit_opinion_t1",
    "audit_nonclean_consec",
    "audit_nonclean_5y",
    "audit_observed",
]


def _parse_opinion(text: str | None) -> int | None:
    """감사의견 텍스트 → int 매핑. 알 수 없으면 None."""
    if not text:
        return None
    t = text.strip()
    if t in OPINION_TO_INT:
        return OPINION_TO_INT[t]
    # 일부 보고서는 '적정의견', '한정의견' 등으로 적힘
    for k, v in OPINION_TO_INT.items():
        if k in t:
            return v
    return None


def load_opinion_for_corp(corp_code: str, year: int) -> int | None:
    """해당 (corp, year)의 감사의견 (int) 반환. 캐시 없거나 데이터 없으면 None."""
    p = CACHE_DIR / f"{corp_code}_{year}.json"
    if not p.exists():
        return None
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    items = data.get("list") or []
    if not items:
        return None
    # 일반적으로 단일 회계연도 = 1건. 여러 건이면 가장 처음 것.
    return _parse_opinion(items[0].get("adt_opinion"))


def build_audit_history(corp_code: str, year_range: tuple[int, int]) -> dict[int, int]:
    """(corp_code)의 (year → opinion_int) 딕셔너리. opinion 없는 연도는 누락."""
    yr_lo, yr_hi = year_range
    out: dict[int, int] = {}
    for y in range(yr_lo, yr_hi + 1):
        op = load_opinion_for_corp(corp_code, y)
        if op is not None:
            out[y] = op
    return out


def _features_for_target(opinion_hist: dict[int, int], target_year: int) -> dict[str, float]:
    """(corp, year=Y) 시점에서의 5개 피처. Y-1 의견을 가장 최근으로 사용."""
    out = {c: np.nan for c in AUDIT_FEATURE_COLS}
    out["audit_observed"] = 0.0

    op_t = opinion_hist.get(target_year - 1)
    if op_t is None:
        return out
    out["audit_opinion_t"] = float(op_t)
    out["audit_observed"] = 1.0

    op_t1 = opinion_hist.get(target_year - 2)
    if op_t1 is not None:
        out["audit_opinion_t1"] = float(op_t1)

    # 연속 비적정 횟수: Y-1부터 거꾸로 가며 0(적정)이 나올 때까지
    consec = 0
    for k in range(1, 6):  # Y-1, Y-2, ..., Y-5
        op = opinion_hist.get(target_year - k)
        if op is None or op == 0:
            break
        consec += 1
    out["audit_nonclean_consec"] = float(consec)

    # Y-5 ~ Y-1 5년간 비적정 횟수
    cnt = sum(1 for k in range(1, 6)
              if (op := opinion_hist.get(target_year - k)) is not None and op != 0)
    out["audit_nonclean_5y"] = float(cnt)
    return out


def build_audit_features_panel(year_range: tuple[int, int] = (2014, 2024)) -> pd.DataFrame:
    """fixed_v1 패널의 모든 (stock, year, quarter)에 감사 피처를 산출."""
    stock_to_corp = build_stock_to_corp()
    stocks = load_panel_stock_codes()

    # panel 키
    base = PROJECT_ROOT / "preprocess" / "data" / "processed_fixed_v1" / "fixed_N1" / "exp-A"
    parts = []
    for split in ["train", "valid", "test"]:
        df = pd.read_csv(base / f"{split}.csv", dtype={"stock_code": str})
        parts.append(df[["stock_code", "year", "quarter"]])
    panel_keys = pd.concat(parts).drop_duplicates().reset_index(drop=True)
    panel_keys["stock_code"] = panel_keys["stock_code"].str.zfill(6)
    print(f"[build] {len(panel_keys)} (stock, year, quarter) keys, "
          f"{panel_keys['stock_code'].nunique()} unique stocks")

    # 종목별 audit 이력 미리 로드
    print(f"[build] loading audit history for {len(stocks)} stocks…")
    audit_hist: dict[str, dict[int, int]] = {}
    t0 = time.time()
    for i, s in enumerate(stocks, start=1):
        c = stock_to_corp.get(s)
        if not c:
            audit_hist[s] = {}
            continue
        audit_hist[s] = build_audit_history(c, year_range)
        if i % 500 == 0 or i == len(stocks):
            print(f"  [{i}/{len(stocks)}] elapsed={time.time()-t0:.1f}s")

    # 피처 행렬
    rows = []
    for _, row in panel_keys.iterrows():
        s = row["stock_code"]
        y = int(row["year"])
        q = row["quarter"]
        feats = _features_for_target(audit_hist.get(s, {}), y)
        rows.append({"stock_code": s, "year": y, "quarter": q, **feats})
    df_audit = pd.DataFrame(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "audit_features.csv"
    df_audit.to_csv(out_path, index=False)
    print(f"\n[saved] {out_path.relative_to(PROJECT_ROOT)}  ({len(df_audit)} rows)")

    # 진단
    print("\n[diagnostics]")
    print("audit_observed 비율:", round(df_audit["audit_observed"].mean(), 3))
    print("audit_opinion_t 분포:")
    print(df_audit["audit_opinion_t"].value_counts(dropna=False).sort_index().to_string())
    print("\naudit_nonclean_consec 분포:")
    print(df_audit["audit_nonclean_consec"].value_counts(dropna=False).sort_index().to_string())

    # 라벨과의 단순 교차 (sanity)
    label_csv = base / "train.csv"
    train_labels = pd.read_csv(label_csv, dtype={"stock_code": str})[["stock_code", "year", "quarter", "label"]]
    train_labels["stock_code"] = train_labels["stock_code"].str.zfill(6)
    merged = df_audit.merge(train_labels, on=["stock_code", "year", "quarter"], how="inner")
    print("\n[train 라벨 vs audit_opinion_t]")
    print(merged.groupby(["label", "audit_opinion_t"]).size().unstack(fill_value=0).to_string())
    print("\n[train 라벨 vs audit_nonclean_consec]")
    print(merged.groupby(["label", "audit_nonclean_consec"]).size().unstack(fill_value=0).to_string())

    return df_audit


if __name__ == "__main__":
    build_audit_features_panel()
