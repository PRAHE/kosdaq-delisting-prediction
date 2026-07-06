"""
A4 새 정보 채널 — 주요사항보고서(pblntf_ty='B') 피처 (조기/말기 분리 + 시점 컷오프).

무필터 원본(`data/major_raw/`)에서 공시를 분류:

  조기 신호(EARLY, 만성 부실 — 상폐 선행, 누수-안전):
    - CB/BW/EB 발행 (희석성 부채조달, 반복 = 만성 자금난)
    - 유상증자 (자본조달 압박)
    - 감자 (자본잠식 신호)
    - 최대주주변경 (지배구조 불안정)
  말기 신호(TERMINAL — 상폐 직전, 누수 위험):
    - 부도발생 / 회생절차개시·파산 / 영업정지

시점 컷오프: (stock, year=Y, quarter=Q) 분기말까지 접수된 공시만 사용.
출력: preprocess/data/major/major_features.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.a1_audit.corp_map import build_stock_to_corp

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "major_raw"
OUT_DIR = PROJECT_ROOT / "preprocess" / "data" / "major"
QUARTER_END_MMDD = {"Q1": 331, "H1": 630, "Q3": 930, "ANNUAL": 1231}

EARLY_FEATURES = [
    "maj_financing_n_3y",      # CB+유증 빈도(3y) — 잦으면 만성 자금난
    "maj_cb_n_3y",             # CB/BW/EB 발행 수(3y) — 희석성 사채
    "maj_financing_any_hist",  # 자금조달 이력
    "maj_ownership_change_hist",
    "maj_capital_reduction_hist",
    "maj_days_since_financing",
]
TERMINAL_FEATURES = ["maj_terminal_any_hist", "maj_terminal_n_5y"]
ALL_FEATURES = EARLY_FEATURES + TERMINAL_FEATURES


def classify(report_nm: str) -> tuple[str | None, bool]:
    """(category, is_early). 우선순위 단일 분류. 해당 없으면 (None, None)."""
    n = report_nm
    if "부도" in n:
        return "DEFAULT", False
    if "회생절차" in n or "파산" in n:
        return "BANKRUPTCY", False
    if "영업정지" in n:
        return "SUSPENSION", False
    if any(k in n for k in ["전환사채", "신주인수권부사채", "교환사채"]):
        return "CB", True
    if "유상증자" in n:
        return "RIGHTS_ISSUE", True
    if "감자" in n:
        return "CAPITAL_REDUCTION", True
    if "최대주주" in n and "변경" in n:
        return "OWNERSHIP_CHANGE", True
    return None, None


def _load_corp_events(corp_code: str) -> list[tuple[int, str, bool]]:
    p = RAW_DIR / f"{corp_code}.json"
    if not p.exists():
        return []
    d = json.load(open(p, encoding="utf-8"))
    out = []
    for it in d.get("disclosures", []):
        dt = it.get("rcept_dt")
        if not dt or not str(dt).isdigit():
            continue
        cat, early = classify(it.get("report_nm", ""))
        if cat is None:
            continue
        out.append((int(dt), cat, early))
    return out


def _days_between(d1: int, d2: int) -> int:
    def to_ord(d):
        return (d // 10000) * 365 + ((d // 100) % 100) * 30 + (d % 100)
    return max(0, to_ord(d2) - to_ord(d1))


def _features_at(events, cutoff_int: int) -> dict:
    f = {c: 0.0 for c in ALL_FEATURES}
    f["maj_days_since_financing"] = 9999.0
    past = [(dt, cat, e) for (dt, cat, e) in events if dt <= cutoff_int]
    if not past:
        return f
    cy, mmdd = cutoff_int // 10000, cutoff_int % 10000
    win3 = (cy - 3) * 10000 + mmdd
    win5 = (cy - 5) * 10000 + mmdd
    financing = [(dt, cat) for (dt, cat, e) in past if cat in ("CB", "RIGHTS_ISSUE")]
    term = [dt for (dt, cat, e) in past if not e]

    if financing:
        f["maj_financing_any_hist"] = 1.0
        f["maj_financing_n_3y"] = float(sum(1 for dt, _ in financing if dt >= win3))
        f["maj_cb_n_3y"] = float(sum(1 for dt, c in financing if c == "CB" and dt >= win3))
        f["maj_days_since_financing"] = float(_days_between(max(dt for dt, _ in financing), cutoff_int))
    f["maj_ownership_change_hist"] = float(any(c == "OWNERSHIP_CHANGE" for _, c, _ in past))
    f["maj_capital_reduction_hist"] = float(any(c == "CAPITAL_REDUCTION" for _, c, _ in past))
    if term:
        f["maj_terminal_any_hist"] = 1.0
        f["maj_terminal_n_5y"] = float(sum(1 for dt in term if dt >= win5))
    return f


def build() -> pd.DataFrame:
    s2c = build_stock_to_corp()
    base = PROJECT_ROOT / "preprocess" / "data" / "processed_fixed_v1" / "fixed_N1" / "exp-A"
    parts = []
    for sp in ["train", "valid", "test"]:
        df = pd.read_csv(base / f"{sp}.csv", dtype={"stock_code": str})
        parts.append(df[["stock_code", "year", "quarter"]])
    keys = pd.concat(parts).drop_duplicates().reset_index(drop=True)
    keys["stock_code"] = keys["stock_code"].str.zfill(6)
    print(f"[build major] {len(keys)} keys, {keys['stock_code'].nunique()} stocks")

    ev_cache: dict[str, list] = {}
    rows = []
    for _, r in keys.iterrows():
        s = r["stock_code"]; y = int(r["year"]); q = r["quarter"]
        corp = s2c.get(s)
        if corp not in ev_cache:
            ev_cache[corp] = _load_corp_events(corp) if corp else []
        cutoff = y * 10000 + QUARTER_END_MMDD.get(q, 1231)
        rows.append({"stock_code": s, "year": y, "quarter": q, **_features_at(ev_cache[corp], cutoff)})
    out = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_DIR / "major_features.csv", index=False)
    print(f"[saved] {(OUT_DIR / 'major_features.csv').relative_to(PROJECT_ROOT)} ({len(out)} rows)")

    # 진단: train 라벨 교차
    labels = pd.read_csv(base / "train.csv", dtype={"stock_code": str})[["stock_code", "year", "quarter", "label"]]
    labels["stock_code"] = labels["stock_code"].str.zfill(6)
    m = out.merge(labels, on=["stock_code", "year", "quarter"], how="inner")
    print("\n[train 라벨 vs maj_financing_any_hist]")
    print(m.groupby(["label", "maj_financing_any_hist"]).size().unstack(fill_value=0).to_string())
    print("\n[train 라벨 vs maj_terminal_any_hist]")
    print(m.groupby(["label", "maj_terminal_any_hist"]).size().unstack(fill_value=0).to_string())
    print("\n[조기 신호 보유율(양성 vs 음성)]")
    for lab in [0, 1]:
        sub = m[m.label == lab]
        print(f"  label={lab}: financing≥1 {(sub['maj_financing_any_hist']>0).mean():.3f}, "
              f"CB 3y≥1 {(sub['maj_cb_n_3y']>0).mean():.3f}, "
              f"감자 {(sub['maj_capital_reduction_hist']>0).mean():.3f}, "
              f"최대주주변경 {(sub['maj_ownership_change_hist']>0).mean():.3f}")
    return out


if __name__ == "__main__":
    build()
