"""
A3 외부 데이터 — KRX 일별 OHLCV 수집.

FinanceDataReader로 fixed_v1 패널에 등장하는 모든 종목코드의 일별 OHLCV를
2014-01-01부터 2025-12-31까지 다운로드하고 종목별 CSV로 캐싱.

사용:
    python -m src.research.a3_market.fetch_ohlcv               # 전체 수집
    python -m src.research.a3_market.fetch_ohlcv --limit 50    # 처음 50종목만
    python -m src.research.a3_market.fetch_ohlcv --skip-existing
"""

from __future__ import annotations

import argparse
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = PROJECT_ROOT / "data" / "market_ohlcv"
START_DATE = "2014-01-01"
END_DATE = "2025-12-31"


def collect_stock_codes() -> list[str]:
    """fixed_v1 train/valid/test에 등장하는 모든 종목코드를 모은다."""
    base = PROJECT_ROOT / "preprocess" / "data" / "processed_fixed_v1" / "fixed_N1" / "exp-A"
    codes: set[str] = set()
    for split in ["train", "valid", "test"]:
        df = pd.read_csv(base / f"{split}.csv", dtype={"stock_code": str})
        codes.update(df["stock_code"].astype(str).str.zfill(6).unique())
    return sorted(codes)


def fetch_single(code: str, retries: int = 2) -> tuple[str, pd.DataFrame | None, str | None]:
    """단일 종목 OHLCV 수집. 성공 시 (code, df, None), 실패 시 (code, None, error_msg)."""
    import FinanceDataReader as fdr
    last_err = None
    for attempt in range(retries + 1):
        try:
            df = fdr.DataReader(code, START_DATE, END_DATE)
            if df is None or len(df) == 0:
                return code, None, "empty"
            df = df.copy()
            df.index = pd.to_datetime(df.index)
            df.index.name = "date"
            return code, df, None
        except Exception as e:
            last_err = str(e)
            if attempt < retries:
                time.sleep(0.5)
    return code, None, last_err


def save_one(code: str, df: pd.DataFrame) -> Path:
    out = CACHE_DIR / f"{code}.csv"
    df.to_csv(out)
    return out


def main(limit: int | None = None, skip_existing: bool = True, n_workers: int = 4) -> None:
    warnings.filterwarnings("ignore")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    codes = collect_stock_codes()
    if limit:
        codes = codes[:limit]

    if skip_existing:
        codes = [c for c in codes if not (CACHE_DIR / f"{c}.csv").exists()]

    print(f"[fetch_ohlcv] target {len(codes)} stocks, cache={CACHE_DIR.relative_to(PROJECT_ROOT)}")
    if not codes:
        print("nothing to do.")
        return

    successes = 0
    failures: list[tuple[str, str]] = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(fetch_single, c): c for c in codes}
        for i, fut in enumerate(as_completed(futures), start=1):
            code, df, err = fut.result()
            if df is not None:
                save_one(code, df)
                successes += 1
            else:
                failures.append((code, err or "unknown"))
            if i % 50 == 0 or i == len(codes):
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                remaining = (len(codes) - i) / rate if rate > 0 else 0
                print(f"  [{i}/{len(codes)}] success={successes} fail={len(failures)} "
                      f"elapsed={elapsed:.1f}s rate={rate:.1f}/s eta={remaining:.0f}s")

    print(f"\n[done] success={successes}, fail={len(failures)}, elapsed={time.time() - t0:.1f}s")
    if failures:
        fail_path = CACHE_DIR / "_failures.csv"
        pd.DataFrame(failures, columns=["stock_code", "error"]).to_csv(fail_path, index=False)
        print(f"  failures saved: {fail_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-skip-existing", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    main(limit=args.limit, skip_existing=not args.no_skip_existing, n_workers=args.workers)
