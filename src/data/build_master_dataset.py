"""
build_master_dataset.py
─────────────────────────────────────────────────────────────────────────────
전처리 파이프라인 Step 1

수행 작업:
  1. raw/healthy, raw/delisted 의 JSON → 재무비율 계산 (전처리 없이 원본 그대로)
  2. 구조적 상폐 / 더미 종목 제거
  3. YoY 증가율 계산 (매출액/순이익/영업이익)
  4. KRX 상장일 기반 비상장 기간 데이터 제거 (옵션, 기본 ON)
  5. 연도별 현황 출력 (Step 2의 split 기준 결정용)
  6. combined_raw.csv 저장

NOTE:
  - clean_data.csv 사용 안 함 (raw 폴더에 전체 원본 JSON이 있으므로)
  - 전처리(결측치 보간, 이상치 클리핑)는 Step 2에서 split 이후에 수행
  - 매출액/순이익/영업이익 증가율은 IS frmtrm 결측 문제로 YoY 방식 사용

수정 이력
─────────
[2025-04-17] _CFS, _OFS suffix 대응 패턴 추가.
[2025-04-17] YoY 증가율 계산 추가.
[2025-04-27] process_folder 병렬처리 추가 (ProcessPoolExecutor).
[2026-04-30] 정합성 처리 통합:
  - (stock_code, year, quarter) dedup 추가
    : 같은 키가 raw 폴더의 섹터 중복 / _CFS·_OFS 변형 때문에 여러 행으로
      만들어지는 문제 (별도 진단 결과 31개 기업, 1,922행 영향).
  - KRX 상장일 기반 비상장 데이터 제거 추가 (--filter-by-listing-date, 기본 ON)
    : 종목코드 재사용으로 신 회사의 데이터에 옛 회사 데이터가 섞여 들어가는
      문제 (036220 오상헬스케어 등 10개 기업, 83행 영향).
    : 상장법인목록.xlsx 파일 없으면 경고만 출력하고 skip.
  - 기존 patch_remove_prelisting_rows.py, patch_dedupe_combined.py는 폐기 대상.
[2026-05-09] raw 폴더에서 중복 파일 704건(다른 섹터 복제 619 + plain/_CFS 공존 85)
              을 직접 삭제 → (code, year, quarter) dedup 로직 제거.
              scripts/cleanup_raw_duplicates.py 참조.

출력:
  data/processed/combined_raw.csv   ← Step 2 입력 (전처리 전 원본)

실행:
  python build_master_dataset.py
  python build_master_dataset.py --workers 8
  python build_master_dataset.py --no-listing-filter   # 상장일 필터 skip
"""

import json
import warnings
import re
import sys
import argparse
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────────────────────
BASE_DIR     = Path(r"C:\kwu\KW0SS_PROJECT\kw0ss_project2\preprocess")
RAW_HEALTHY  = BASE_DIR / "data" / "raw" / "healthy"
RAW_DELISTED = BASE_DIR / "data" / "raw" / "delisted"
OUT_DIR      = BASE_DIR / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# KRX 상장법인목록 (선택적 의존성)
LISTED_XLSX = BASE_DIR / "data" / "상장법인목록.xlsx"

# 제거 대상: 구조적 상폐 4개 + 더미
EXCLUDE_CODES = {
    "048260",  # 오스템임플란트 (자진상폐)
    "029960",  # 코엔텍 (완전자회사)
    "006580",  # 대양제지 (자진상폐)
    "115960",  # 연우 (완전자회사)
    "999999",  # 더미
}

PATTERN = re.compile(r"^(\d{6})_(\d{4})_(Q1|Q3|H1|ANNUAL)(?:_CFS|_OFS)?\.json$")

YOY_TARGETS: list[tuple[str, str]] = [
    ("매출액증가율",   "revenue"),
    ("순이익증가율",   "net_income"),
    ("영업이익증가율", "operating_income"),
]
YOY_SOURCE_KEYS = [feat for _, feat in YOY_TARGETS]

# 분기 → 분기 마지막 날짜 (상장일 비교용)
QUARTER_END_MONTH_DAY = {
    "Q1":     (3, 31),
    "H1":     (6, 30),
    "Q3":     (9, 30),
    "ANNUAL": (12, 31),
}


# ─────────────────────────────────────────────────────────────
# 1. 단일 파일 처리 (병렬 worker용)
# ─────────────────────────────────────────────────────────────
def process_file(args: tuple) -> dict | None:
    fp, label, base_dir = args

    sys.path.insert(0, str(base_dir / "src"))
    from account_mapper import extract_standard_items
    from ratio_calculator import compute_all_ratios

    m = PATTERN.match(fp.name)
    if not m:
        return None

    code    = m.group(1).zfill(6)
    year    = int(m.group(2))
    quarter = m.group(3)
    sector  = fp.parent.name

    if code in EXCLUDE_CODES:
        return None

    try:
        with open(fp, encoding="utf-8") as f:
            dart_items = json.load(f)
    except Exception:
        return None

    if not dart_items:
        return None

    std_items = extract_standard_items(dart_items)
    ratios    = compute_all_ratios(std_items)

    record = {
        "stock_code":  code,
        "year":        year,
        "quarter":     quarter,
        "label":       label,
        "gics_sector": sector,
    }
    record.update(ratios)

    for key in YOY_SOURCE_KEYS:
        entry = std_items.get(key)
        record[f"_yoy_src_{key}"] = entry.get("thstrm") if entry else None

    return record


# ─────────────────────────────────────────────────────────────
# 2. 폴더 병렬 처리
# ─────────────────────────────────────────────────────────────
def process_folder(folder: Path, label: int, workers: int) -> pd.DataFrame:
    files  = list(folder.rglob("*.json"))
    split  = "healthy" if label == 0 else "delisted"
    print(f"  {split}: {len(files):,}개 파일 (workers={workers})")

    args_list = [(fp, label, BASE_DIR) for fp in files]
    records   = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_file, a): a for a in args_list}
        for future in tqdm(
            as_completed(futures),
            total=len(files),
            desc=f"  {split}",
            leave=False,
        ):
            result = future.result()
            if result is not None:
                records.append(result)

    df = pd.DataFrame(records)
    print(f"  {split}: {len(df):,}행 변환 완료")
    return df


# ─────────────────────────────────────────────────────────────
# 3. YoY 증가율 계산 (기존)
# ─────────────────────────────────────────────────────────────
def _add_yoy_growth_cols(df: pd.DataFrame) -> pd.DataFrame:
    """전년 동기 조인 방식으로 IS 증가율 3개 계산."""
    df = df.copy()
    df["_prev_year"] = df["year"] - 1

    src_cols = [f"_yoy_src_{feat}" for _, feat in YOY_TARGETS]

    prev_df = df[["stock_code", "year", "quarter"] + src_cols].copy()
    df = df.merge(
        prev_df,
        left_on=["stock_code", "_prev_year", "quarter"],
        right_on=["stock_code", "year", "quarter"],
        suffixes=("", "_prev"),
        how="left",
    )
    df = df.drop(columns=["year_prev", "quarter_prev"], errors="ignore")

    for col_name, feat in YOY_TARGETS:
        src      = f"_yoy_src_{feat}"
        src_prev = f"_yoy_src_{feat}_prev"
        if src not in df.columns or src_prev not in df.columns:
            continue
        denom = df[src_prev].abs()
        result = (df[src] - df[src_prev]) / denom * 100
        result[denom == 0] = np.nan
        df[col_name] = result

    drop_cols = (
        ["_prev_year"]
        + src_cols
        + [f"{c}_prev" for c in src_cols]
    )
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df


# ─────────────────────────────────────────────────────────────
# 4. 상장일 필터링
# ─────────────────────────────────────────────────────────────
def _filter_by_listing_date(
    df: pd.DataFrame,
    listed_xlsx: Path,
) -> pd.DataFrame:
    """
    KRX 상장법인목록을 사용해 각 기업의 상장일 이전 분기 데이터 제거.

    종목코드 재사용 케이스 (같은 코드가 폐지된 다른 회사에 재할당)에서
    신 회사의 raw 데이터에 옛 회사 데이터가 섞여 들어가는 문제 해결.

    Parameters
    ----------
    df : pd.DataFrame
        combined raw DataFrame
    listed_xlsx : Path
        KRX 상장법인목록.xlsx 경로

    Returns
    -------
    pd.DataFrame
        상장일 이전 행이 제거된 DataFrame.
        파일이 없거나 매핑이 없는 기업은 그대로 유지.
    """
    if not listed_xlsx.exists():
        print(f"  ⚠️  {listed_xlsx} 없음 → 상장일 필터링 skip")
        print(f"     (data/ 폴더에 KRX 상장법인목록.xlsx 다운로드 권장)")
        return df

    listed = pd.read_excel(listed_xlsx)
    if "종목코드" not in listed.columns or "상장일" not in listed.columns:
        print(f"  ⚠️  {listed_xlsx}에 '종목코드' 또는 '상장일' 컬럼 없음 → skip")
        return df

    listed["종목코드"] = listed["종목코드"].astype(str).str.strip()
    # 6자리 숫자만 (스팩, ETF 등 제외)
    listed = listed[listed["종목코드"].str.match(r"^\d{6}$")]
    listed["상장일"] = pd.to_datetime(listed["상장일"], errors="coerce")
    list_dt_map = dict(zip(listed["종목코드"], listed["상장일"]))
    print(f"  상장일 매핑: {len(list_dt_map):,}개 기업")

    # 분기 → quarter_end 계산
    def quarter_end(year: int, quarter: str) -> pd.Timestamp:
        if quarter not in QUARTER_END_MONTH_DAY:
            return pd.NaT
        m, d = QUARTER_END_MONTH_DAY[quarter]
        return pd.Timestamp(year=year, month=m, day=d)

    df = df.copy()
    df["_quarter_end"] = df.apply(
        lambda r: quarter_end(int(r["year"]), r["quarter"]),
        axis=1,
    )

    # 각 행에 대응하는 상장일
    df["_list_dt"] = df["stock_code"].map(list_dt_map)

    # 제거 조건: 상장일 매핑이 있고 + quarter_end < 상장일
    mask_remove = (
        df["_list_dt"].notna()
        & df["_quarter_end"].notna()
        & (df["_quarter_end"] < df["_list_dt"])
    )

    n_remove = int(mask_remove.sum())
    if n_remove == 0:
        print(f"  상장일 이전 행 없음")
        df = df.drop(columns=["_quarter_end", "_list_dt"])
        return df

    # 영향받는 기업 통계
    affected = df[mask_remove].groupby("stock_code").agg(
        n_removed=("year", "count"),
        list_dt=("_list_dt", "first"),
        first_year=("year", "min"),
        last_year=("year", "max"),
    ).reset_index()

    print(f"  상장일 이전 행 {n_remove:,}개 제거 ({len(affected)}개 기업):")
    for _, r in affected.iterrows():
        list_str = str(r["list_dt"].date()) if pd.notna(r["list_dt"]) else "?"
        print(f"    {r['stock_code']:<8} 상장일 {list_str} "
              f"({int(r['first_year'])}~{int(r['last_year'])}, "
              f"{int(r['n_removed'])}행 제거)")

    df_clean = df[~mask_remove].drop(columns=["_quarter_end", "_list_dt"])
    df_clean = df_clean.reset_index(drop=True)
    return df_clean


# ─────────────────────────────────────────────────────────────
# 5. 연도별 현황 출력 (기존)
# ─────────────────────────────────────────────────────────────
def print_yearly_stats(df: pd.DataFrame):
    print("\n" + "=" * 65)
    print("  연도별 현황 — Step 2 split 기준 결정에 활용하세요")
    print("=" * 65)
    print(f"\n  {'연도':<6} {'전체행':>8} {'기업수':>7} {'상폐기업수':>10} {'상폐행수':>9}")
    print("  " + "-" * 44)

    for year, grp in df.groupby("year"):
        total         = len(grp)
        companies     = grp["stock_code"].nunique()
        pos_companies = grp[grp["label"] == 1]["stock_code"].nunique()
        pos_rows      = int(grp["label"].sum())
        flag = "  ← 양성 10개 미만" if pos_rows < 10 else ""
        print(f"  {int(year):<6} {total:>8,} {companies:>7,} "
              f"{pos_companies:>10} {pos_rows:>9}{flag}")

    total_pos = int(df["label"].sum())
    total_neg = int((df["label"] == 0).sum())
    print(f"\n  전체: {len(df):,}행 | "
          f"정상 {total_neg:,} : 상폐 {total_pos} "
          f"= {total_neg // max(total_pos, 1)}:1")
    print("\n  [참고] H별 라벨링 시 양성 샘플 수가 달라집니다.")
    print("  Step 2에서 H값에 따라 split 연도를 결정하세요.")
    print("  일반적으로: valid/test 각각 양성 20개 이상 확보 권장")


# ─────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Step 1: JSON → 재무비율 계산 + YoY 증가율 + 상장일 필터"
    )
    parser.add_argument(
        "--workers", type=int,
        default=max(1, os.cpu_count() - 1),
        help="병렬 프로세스 수 (기본: CPU 코어 수 - 1)"
    )
    parser.add_argument(
        "--no-listing-filter", action="store_true",
        help="KRX 상장일 기반 필터링 skip (기본: 적용)"
    )
    args = parser.parse_args()

    print("=" * 65)
    print("  Step 1: JSON → 재무비율 계산 (raw 폴더 전체)")
    print(f"  workers: {args.workers} / CPU: {os.cpu_count()}")
    print(f"  상장일 필터: {'OFF' if args.no_listing_filter else 'ON'}")
    print("=" * 65)

    # [1/4] JSON → 재무비율 (병렬)
    print("\n[1/4] raw 데이터 병렬 변환 중...")
    healthy  = process_folder(RAW_HEALTHY,  label=0, workers=args.workers)
    delisted = process_folder(RAW_DELISTED, label=1, workers=args.workers)

    combined = pd.concat([healthy, delisted], ignore_index=True)
    combined = combined.sort_values(
        ["stock_code", "year", "quarter"]
    ).reset_index(drop=True)
    print(f"\n  변환 완료: {len(combined):,}행 "
          f"(healthy {len(healthy):,} + delisted {len(delisted):,})")
    print(f"  기업 수: {combined['stock_code'].nunique():,}")

    # [2/4] YoY 증가율 계산
    print("\n[2/4] YoY 증가율 계산 중...")
    combined = _add_yoy_growth_cols(combined)
    for col_name, _ in YOY_TARGETS:
        null_rate = combined[col_name].isna().mean()
        print(f"  {col_name} 결측률: {null_rate:.1%}")

    # [3/4] 상장일 필터링
    if not args.no_listing_filter:
        print("\n[3/4] KRX 상장일 기반 비상장 기간 데이터 제거...")
        combined = _filter_by_listing_date(combined, LISTED_XLSX)
    else:
        print("\n[3/4] 상장일 필터링 skip (--no-listing-filter)")

    # 저장
    out_path = OUT_DIR / "combined_raw.csv"
    combined.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n  저장 완료 → {out_path}")
    print(f"  최종: {len(combined):,}행, {combined['stock_code'].nunique():,}개 기업")

    # [4/4] 연도별 현황 출력
    print("\n[4/4] 연도별 현황 확인...")
    print_yearly_stats(combined)

    print("\n" + "=" * 65)
    print("  Step 1 완료")
    print("  다음: 연도별 현황 확인 후 H값과 split 기준 결정")
    print("        → build_h_datasets.py 실행")
    print("=" * 65)


if __name__ == "__main__":
    main()