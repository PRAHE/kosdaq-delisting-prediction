"""
A1 외부 데이터 — DART OpenAPI 감사의견 수집.

엔드포인트:
  https://opendart.fss.or.kr/api/accnutAdtorNmNdAdtOpinion.json
  파라미터: crtfc_key, corp_code (8자리), bsns_year (YYYY), reprt_code

reprt_code:
  11011 — 사업보고서 (연간, 감사의견 핵심)
  11013 — 1분기보고서  11012 — 반기  11014 — 3분기

응답 핵심 필드:
  adt_opinion           — 감사의견 ("적정", "한정", "부적정", "의견거절")
  emphs_matter          — 강조사항 (계속기업 불확실성 등 텍스트)
  adt_reprt_spcmnt_matter — 감사보고서 특기사항

사용:
  python -m src.research.a1_audit.fetch_audit                    # 전체
  python -m src.research.a1_audit.fetch_audit --limit 30          # 30종목만
  python -m src.research.a1_audit.fetch_audit --years 2015 2024   # 특정 연도
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from dotenv import load_dotenv

from src.research.a1_audit.corp_map import build_stock_to_corp, load_panel_stock_codes

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = PROJECT_ROOT / "data" / "audit_opinion"
API_URL = "https://opendart.fss.or.kr/api/accnutAdtorNmNdAdtOpinion.json"
REPRT_CODE_ANNUAL = "11011"

DEFAULT_YEAR_RANGE = (2014, 2024)  # train 2015~, history는 한 해 전부터


def _cache_path(corp_code: str, year: int) -> Path:
    return CACHE_DIR / f"{corp_code}_{year}.json"


def fetch_one(
    api_key: str,
    corp_code: str,
    year: int,
    reprt_code: str = REPRT_CODE_ANNUAL,
    retries: int = 2,
    timeout: int = 30,
) -> tuple[str, int, dict | None, str | None]:
    """단일 (corp_code, year) 감사의견 호출. 성공 시 (corp, year, json, None)."""
    params = dict(
        crtfc_key=api_key,
        corp_code=corp_code,
        bsns_year=str(year),
        reprt_code=reprt_code,
    )
    last_err = None
    for attempt in range(retries + 1):
        try:
            r = requests.get(API_URL, params=params, timeout=timeout)
            if r.status_code != 200:
                last_err = f"http_{r.status_code}"
            else:
                data = r.json()
                status = data.get("status", "")
                if status not in ("000", "013"):  # 013 = 조회된 데이터 없음 (정상)
                    last_err = f"api_status_{status}: {data.get('message','')}"
                else:
                    return corp_code, year, data, None
        except Exception as e:
            last_err = str(e)
        if attempt < retries:
            time.sleep(0.5)
    return corp_code, year, None, last_err


def save(corp_code: str, year: int, data: dict) -> Path:
    p = _cache_path(corp_code, year)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return p


def main(
    year_range: tuple[int, int] = DEFAULT_YEAR_RANGE,
    limit: int | None = None,
    skip_existing: bool = True,
    n_workers: int = 4,
) -> None:
    warnings.filterwarnings("ignore")
    load_dotenv()
    api_key = os.getenv("DART_API_KEY", "").strip()
    if not api_key:
        raise SystemExit(
            "DART_API_KEY가 .env에 없습니다. "
            "https://opendart.fss.or.kr/uss/umt/login/loginView.do 에서 발급 후 "
            "프로젝트 루트의 .env에 DART_API_KEY=... 형태로 저장하세요."
        )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    stock_to_corp = build_stock_to_corp()
    stocks = load_panel_stock_codes()
    if limit:
        stocks = stocks[:limit]

    yr_lo, yr_hi = year_range
    years = list(range(yr_lo, yr_hi + 1))

    # 작업 단위: (corp_code, year) — 캐시 있으면 skip
    tasks: list[tuple[str, int]] = []
    skipped = 0
    for s in stocks:
        c = stock_to_corp.get(s)
        if not c:
            continue
        for y in years:
            if skip_existing and _cache_path(c, y).exists():
                skipped += 1
                continue
            tasks.append((c, y))

    print(f"[fetch_audit] target {len(tasks)} (corp, year) calls "
          f"({len(stocks)} stocks × {len(years)} years), skipped {skipped} cached")
    if not tasks:
        print("nothing to do.")
        return

    successes = 0
    failures: list[tuple[str, int, str]] = []
    t0 = time.time()

    # DART API rate limit ~10 calls/sec — too many parallel risks throttling
    n_workers = min(n_workers, 4)

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(fetch_one, api_key, c, y): (c, y) for c, y in tasks}
        for i, fut in enumerate(as_completed(futures), start=1):
            corp, year, data, err = fut.result()
            if data is not None:
                save(corp, year, data)
                successes += 1
            else:
                failures.append((corp, year, err or "unknown"))
            if i % 100 == 0 or i == len(tasks):
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(tasks) - i) / rate if rate > 0 else 0
                print(f"  [{i}/{len(tasks)}] success={successes} fail={len(failures)} "
                      f"elapsed={elapsed:.1f}s rate={rate:.1f}/s eta={eta:.0f}s")

    print(f"\n[done] success={successes}, fail={len(failures)}, elapsed={time.time()-t0:.1f}s")
    if failures:
        import pandas as pd
        fail_path = CACHE_DIR / "_failures.csv"
        pd.DataFrame(failures, columns=["corp_code", "year", "error"]).to_csv(fail_path, index=False)
        print(f"  failures saved: {fail_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--years", nargs=2, type=int, default=list(DEFAULT_YEAR_RANGE),
                     metavar=("FROM", "TO"))
    ap.add_argument("--no-skip-existing", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    main(year_range=tuple(args.years), limit=args.limit,
         skip_existing=not args.no_skip_existing, n_workers=args.workers)
