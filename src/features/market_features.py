"""
A3 외부 데이터 — 분기별 시장 피처 집계.

캐시된 일별 OHLCV에서 (stock_code, year, quarter) 단위 6개 피처를 만든다.
분기 시점 매핑:
  ANNUAL → 12월 마지막 거래일
  Q1     → 3월 마지막 거래일
  H1     → 6월 마지막 거래일
  Q3     → 9월 마지막 거래일

피처:
  price_log_close          — 시점 종가 로그
  price_ret_12m            — 12개월 수익률 (시점 종가 / 1년 전 종가 - 1)
  price_volatility_60d     — 직전 60거래일 일일수익률 std × √252 (annualized)
  price_drawdown_max_12m   — (12m 최고가 - 현재가) / 12m 최고가
  volume_log_mean_60d      — 직전 60거래일 평균 거래량 로그
  volume_change_yoy        — 직전 60일 평균 거래량 / 1년 전 동기간 평균 - 1
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = PROJECT_ROOT / "data" / "market_ohlcv"
OUT_DIR = PROJECT_ROOT / "preprocess" / "data" / "market"

QUARTER_TO_MONTH = {"Q1": 3, "H1": 6, "Q3": 9, "ANNUAL": 12}

MARKET_FEATURE_COLS = [
    "price_log_close",
    "price_ret_12m",
    "price_volatility_60d",
    "price_drawdown_max_12m",
    "volume_log_mean_60d",
    "volume_change_yoy",
]


def _last_business_day_of_month(df_index: pd.DatetimeIndex, year: int, month: int) -> pd.Timestamp | None:
    """df_index에서 (year, month)의 마지막 거래일 찾기. 없으면 None."""
    mask = (df_index.year == year) & (df_index.month == month)
    if not mask.any():
        return None
    return df_index[mask].max()


def _compute_features_at(df: pd.DataFrame, t: pd.Timestamp) -> dict[str, float]:
    """df는 (date index, Close, Volume 컬럼)을 가진다. t시점 기준 6개 피처 계산."""
    # 인덱스가 정렬되어 있다고 가정
    idx = df.index
    # t에 해당하는 row의 위치
    pos = df.index.searchsorted(t, side="right") - 1
    if pos < 0 or pos >= len(df):
        return {c: np.nan for c in MARKET_FEATURE_COLS}

    close_t = float(df["Close"].iloc[pos])
    if close_t <= 0:
        return {c: np.nan for c in MARKET_FEATURE_COLS}

    # 직전 60거래일 윈도우
    lo60 = max(0, pos - 59)
    win60 = df.iloc[lo60: pos + 1]
    # 1년 전 시점 (대략 252거래일 전)
    pos_1y = pos - 252
    close_1y = float(df["Close"].iloc[pos_1y]) if pos_1y >= 0 else np.nan
    # 1년 전 60일 윈도우 (대략 252~312거래일 전)
    lo_1y = max(0, pos_1y - 59)
    win60_1y = df.iloc[lo_1y: pos_1y + 1] if pos_1y >= 0 else pd.DataFrame()

    # price_log_close
    price_log_close = float(np.log1p(close_t))

    # price_ret_12m
    if np.isfinite(close_1y) and close_1y > 0:
        price_ret_12m = close_t / close_1y - 1.0
    else:
        price_ret_12m = np.nan

    # price_volatility_60d (annualized)
    if len(win60) >= 10:
        log_ret = np.log(win60["Close"].clip(lower=1e-9)).diff().dropna()
        if len(log_ret) >= 5:
            price_volatility_60d = float(log_ret.std() * np.sqrt(252))
        else:
            price_volatility_60d = np.nan
    else:
        price_volatility_60d = np.nan

    # price_drawdown_max_12m: 직전 252거래일 최고가 대비 낙폭
    lo_12m = max(0, pos - 251)
    win_12m = df.iloc[lo_12m: pos + 1]
    high_12m = float(win_12m["High"].max()) if "High" in df.columns else float(win_12m["Close"].max())
    price_drawdown_max_12m = (high_12m - close_t) / high_12m if high_12m > 0 else np.nan

    # volume_log_mean_60d
    if len(win60) >= 5 and "Volume" in df.columns:
        vol_mean = float(win60["Volume"].mean())
        volume_log_mean_60d = float(np.log1p(max(vol_mean, 0)))
    else:
        volume_log_mean_60d = np.nan

    # volume_change_yoy
    if "Volume" in df.columns and len(win60) >= 5 and len(win60_1y) >= 5:
        v_now = float(win60["Volume"].mean())
        v_old = float(win60_1y["Volume"].mean())
        if v_old > 0:
            volume_change_yoy = v_now / v_old - 1.0
        else:
            volume_change_yoy = np.nan
    else:
        volume_change_yoy = np.nan

    return {
        "price_log_close":         price_log_close,
        "price_ret_12m":           price_ret_12m,
        "price_volatility_60d":    price_volatility_60d,
        "price_drawdown_max_12m":  price_drawdown_max_12m,
        "volume_log_mean_60d":     volume_log_mean_60d,
        "volume_change_yoy":       volume_change_yoy,
    }


def build_features_for_stock(code: str, year_quarters: list[tuple[int, str]]) -> pd.DataFrame:
    """단일 종목의 (year, quarter) 리스트에 대해 시장 피처 산출."""
    path = CACHE_DIR / f"{code}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    df = df.sort_index()
    rows = []
    for year, quarter in year_quarters:
        month = QUARTER_TO_MONTH.get(quarter)
        if month is None:
            continue
        t = _last_business_day_of_month(df.index, year, month)
        if t is None:
            rows.append({"stock_code": code, "year": year, "quarter": quarter,
                         **{c: np.nan for c in MARKET_FEATURE_COLS}})
            continue
        feats = _compute_features_at(df, t)
        rows.append({"stock_code": code, "year": year, "quarter": quarter, **feats})
    return pd.DataFrame(rows)


def build_market_features_panel() -> pd.DataFrame:
    """fixed_v1 panel의 모든 (stock_code, year, quarter)에 대해 시장 피처 산출."""
    from src.research.s1_irregular_ts.sequences_grud import _load_combined_with_macro  # reuse
    # We don't actually need combined_raw; just need the (stock,year,quarter) keys from fixed_v1
    base = PROJECT_ROOT / "preprocess" / "data" / "processed_fixed_v1" / "fixed_N1" / "exp-A"
    keys = []
    for split in ["train", "valid", "test"]:
        df = pd.read_csv(base / f"{split}.csv", dtype={"stock_code": str})
        keys.append(df[["stock_code", "year", "quarter"]])
    panel_keys = pd.concat(keys).drop_duplicates().reset_index(drop=True)

    panel_keys["stock_code"] = panel_keys["stock_code"].str.zfill(6)
    print(f"[build] {len(panel_keys)} (stock, year, quarter) keys, "
          f"{panel_keys['stock_code'].nunique()} unique stocks")

    all_rows = []
    by_stock = panel_keys.groupby("stock_code")
    n = len(by_stock)
    import time
    t0 = time.time()
    for i, (code, grp) in enumerate(by_stock, start=1):
        yq = list(zip(grp["year"].astype(int), grp["quarter"]))
        out = build_features_for_stock(code, yq)
        if not out.empty:
            all_rows.append(out)
        if i % 200 == 0 or i == n:
            elapsed = time.time() - t0
            print(f"  [{i}/{n}] elapsed={elapsed:.1f}s")

    df_market = pd.concat(all_rows, ignore_index=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "market_features.csv"
    df_market.to_csv(out_path, index=False)
    print(f"\n[saved] {out_path.relative_to(PROJECT_ROOT)}  ({len(df_market)} rows)")

    # 진단
    print("\n[diagnostics]")
    obs_ratio = df_market[MARKET_FEATURE_COLS].notna().mean()
    print("obs_ratio per feature:")
    print(obs_ratio.round(3).to_string())
    return df_market


if __name__ == "__main__":
    build_market_features_panel()
