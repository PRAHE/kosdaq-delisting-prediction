"""DART raw JSON → 재무비율 CSV 변환 모듈.

account_mapper + ratio_calculator 를 조합하여
raw JSON 파일을 재무비율 CSV로 변환한다.

사용법:
    # 단일 기업
    python -m preprocess.src.etl single \
        --raw-dir data/raw/sample/healthy/Materials \
        --ticker 001810 --year 2025 --corp-name 무림SP --label 0 \
        --output data/output/sample/Materials/001810_2025.csv

    # 디렉터리 일괄 변환
    python -m preprocess.src.etl batch \
        --raw-base data/raw/sample \
        --output-base data/output/sample \
        --company-csv companies.csv
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Optional

from src.data.account_mapper import extract_standard_items
from src.data.ratio_calculator import RATIO_NAMES, compute_all_ratios

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------

QUARTER_ORDER = ["Q1", "H1", "Q3", "ANNUAL"]

META_COLS = ["stock_code", "corp_name", "year", "quarter", "label"]

ALL_COLS = META_COLS + RATIO_NAMES


# ---------------------------------------------------------------------------
# CSV 생성
# ---------------------------------------------------------------------------


def convert_single(
    ticker: str,
    corp_name: str,
    year: int,
    label: int,
    raw_dir: str | Path,
    output_path: str | Path,
) -> int:
    """단일 기업/연도의 raw JSON → 재무비율 CSV 변환.

    Returns:
        생성된 행 수
    """
    raw_dir = Path(raw_dir)
    output_path = Path(output_path)
    rows: list[dict[str, str]] = []

    for quarter in QUARTER_ORDER:
        fpath = raw_dir / f"{ticker}_{year}_{quarter}.json"
        if not fpath.exists():
            continue

        with open(fpath, encoding="utf-8") as f:
            data = json.load(f)

        # account_mapper → ratio_calculator 파이프라인
        items = extract_standard_items(data)
        ratios = compute_all_ratios(items)

        row = {
            "stock_code": ticker,
            "corp_name": corp_name,
            "year": str(year),
            "quarter": quarter,
            "label": str(label),
        }
        for col in RATIO_NAMES:
            val = ratios.get(col)
            row[col] = str(val) if val is not None else ""
        rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=ALL_COLS)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def convert_batch(
    raw_base: str | Path,
    output_base: str | Path,
    company_map: dict[str, dict],
) -> list[str]:
    """raw_base 하위의 모든 JSON을 일괄 변환.

    Args:
        raw_base: data/raw/ 루트 (하위: {status}/{sector}/{ticker}_{year}_{quarter}.json)
        output_base: data/output/ 루트
        company_map: ticker → {"corp_name": str, "label": int, "sector": str}

    Returns:
        생성된 CSV 경로 리스트
    """
    raw_base = Path(raw_base)
    output_base = Path(output_base)
    generated = []

    # ticker_year 조합 수집
    seen: dict[tuple[str, str, str, str], Path] = {}
    for json_path in sorted(raw_base.rglob("*.json")):
        parts = json_path.relative_to(raw_base).parts
        if len(parts) < 3:
            continue
        status, sector = parts[0], parts[1]
        fname = json_path.stem
        tokens = fname.split("_")
        if len(tokens) < 3:
            continue
        ticker, year = tokens[0], tokens[1]
        key = (status, sector, ticker, year)
        if key not in seen:
            seen[key] = json_path.parent

    for (status, sector, ticker, year), raw_dir in sorted(seen.items()):
        info = company_map.get(ticker, {})
        corp_name = info.get("corp_name", "")
        label = info.get("label", 1 if status == "delisted" else 0)

        out_path = output_base / sector / f"{ticker}_{year}.csv"
        n = convert_single(ticker, corp_name, int(year), label, raw_dir, out_path)
        if n > 0:
            generated.append(str(out_path))
            print(f"  {out_path} ({n} rows)")

    return generated


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    import argparse

    parser = argparse.ArgumentParser(description="DART raw JSON → 재무비율 CSV")
    sub = parser.add_subparsers(dest="cmd")

    # -- single --
    p_single = sub.add_parser("single", help="단일 기업 변환")
    p_single.add_argument("--raw-dir", required=True, help="raw JSON 디렉터리")
    p_single.add_argument("--ticker", required=True)
    p_single.add_argument("--year", type=int, required=True)
    p_single.add_argument("--corp-name", default="")
    p_single.add_argument("--label", type=int, default=0)
    p_single.add_argument("--output", required=True, help="출력 CSV 경로")

    # -- batch --
    p_batch = sub.add_parser("batch", help="일괄 변환")
    p_batch.add_argument("--raw-base", required=True, help="raw 루트 (e.g. data/raw)")
    p_batch.add_argument("--output-base", required=True, help="output 루트 (e.g. data/output)")
    p_batch.add_argument("--company-csv", default=None, help="기업 매핑 CSV (stock_code,corp_name)")

    args = parser.parse_args()

    if args.cmd == "single":
        n = convert_single(
            args.ticker, args.corp_name, args.year, args.label,
            args.raw_dir, args.output,
        )
        print(f"Generated {n} rows → {args.output}")

    elif args.cmd == "batch":
        company_map: dict[str, dict] = {}
        if args.company_csv and os.path.exists(args.company_csv):
            with open(args.company_csv, encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    company_map[row["stock_code"]] = {
                        "corp_name": row.get("corp_name", ""),
                        "label": int(row.get("label", 0)),
                        "sector": row.get("gics_sector", ""),
                    }
        results = convert_batch(args.raw_base, args.output_base, company_map)
        print(f"\nTotal: {len(results)} files generated")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
