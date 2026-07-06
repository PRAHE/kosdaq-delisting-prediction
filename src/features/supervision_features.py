"""
A2 재평가 — supervision v2 피처 (확장 키워드 + 조기/말기 분리 + 시점 컷오프).

#7 감사 결과: 기존 키워드가 상폐 파이프라인 공시를 전부 놓쳤다. 본 빌더는
무필터 원본(`data/supervision_raw/`)에서 공시를 분류하고, **누수 통제**를 위해
공시를 두 부류로 나눈다:

  조기 신호(EARLY, 상폐보다 충분히 앞섬 — 안전):
    - SUBSTANTIVE_REVIEW : 상장적격성 실질심사 대상/사유 (보통 1년+ 전)
    - MANAGEMENT         : 관리종목 지정/우려/내부결산
    - UNFAITHFUL         : 불성실공시법인 지정
    - CAUTION            : 투자주의/경고/위험/단기과열
  말기 신호(TERMINAL, 상폐 직전/당시 — 누수 위험):
    - DELIST_CAUSE       : 상장폐지 사유 발생/관련/결정
    - LIQUIDATION        : 정리매매 개시

시점 컷오프: (stock, year=Y, quarter=Q)의 분기말까지 접수된 공시만 사용
  (rcept_dt ≤ 분기말). 미래 공시 미사용.

출력: preprocess/data/supervision_v2/supervision_features_v2.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.a1_audit.corp_map import build_stock_to_corp

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "supervision_raw"
OUT_DIR = PROJECT_ROOT / "preprocess" / "data" / "supervision_v2"
QUARTER_END_MMDD = {"Q1": 331, "H1": 630, "Q3": 930, "ANNUAL": 1231}

EARLY_FEATURES = [
    "sup2_early_any_hist", "sup2_early_n_5y", "sup2_days_since_early",
    "sup2_review_any_hist", "sup2_unfaithful_5y", "sup2_mgmt_5y",
]
TERMINAL_FEATURES = ["sup2_terminal_any_hist", "sup2_terminal_n_5y"]
ALL_FEATURES = EARLY_FEATURES + TERMINAL_FEATURES


def classify(report_nm: str) -> tuple[str | None, bool]:
    """(category, is_early). 우선순위로 단일 분류.

    주의: '내부결산시점관리종목지정ㆍ형식적상장폐지ㆍ실질심사…' 복합공시는
    가장 흔한 유형이자 내부결산(상폐 ~1년 전) 시점의 **조기 복합 경고**이므로
    '상장폐지' 글자가 있어도 MANAGEMENT(early)로 분류한다. 시점 컷오프가
    과거 공시만 사용하도록 보장하므로 안전하다.
    """
    n = report_nm
    if "정리매매" in n:                                   # 명백한 말기
        return "LIQUIDATION", False
    if "내부결산시점" in n:                                # 조기 복합 경고(최다 유형)
        return "MANAGEMENT", True
    if "실질심사" in n or "상장적격성" in n:                # 실질심사 대상/사유 = 조기
        return "SUBSTANTIVE_REVIEW", True
    if "불성실공시" in n:
        return "UNFAITHFUL", True
    if "관리종목" in n:
        return "MANAGEMENT", True
    if "상장폐지" in n:                                    # 위에서 안 걸린 상폐 = 말기(결정/이의신청/여부)
        return "DELIST_CAUSE", False
    if any(k in n for k in ["투자주의", "투자경고", "투자위험", "단기과열"]):
        return "CAUTION", True
    return None, False                                     # 풍문 등 일반 거래정지/기타는 제외


def _load_corp_events(corp_code: str) -> list[tuple[int, str, bool]]:
    """corp의 (rcept_dt_int, category, is_early) 리스트."""
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


def _features_at(events, cutoff_int: int) -> dict:
    """cutoff(YYYYMMDD int)까지의 공시로 피처 계산."""
    f = {c: 0.0 for c in ALL_FEATURES}
    f["sup2_days_since_early"] = 9999.0
    past = [(dt, cat, e) for (dt, cat, e) in events if dt <= cutoff_int]
    if not past:
        return f
    cy = cutoff_int // 10000
    mmdd = cutoff_int % 10000
    win5 = (cy - 5) * 10000 + mmdd
    early = [(dt, cat) for (dt, cat, e) in past if e]
    term = [(dt, cat) for (dt, cat, e) in past if not e]

    if early:
        f["sup2_early_any_hist"] = 1.0
        f["sup2_early_n_5y"] = float(sum(1 for dt, _ in early if dt >= win5))
        last = max(dt for dt, _ in early)
        # 근사 일수: YYYYMMDD 차이를 일수로 환산
        f["sup2_days_since_early"] = float(_days_between(last, cutoff_int))
        f["sup2_review_any_hist"] = float(any(c == "SUBSTANTIVE_REVIEW" for _, c in early))
        f["sup2_unfaithful_5y"] = float(sum(1 for dt, c in early if c == "UNFAITHFUL" and dt >= win5))
        f["sup2_mgmt_5y"] = float(sum(1 for dt, c in early if c == "MANAGEMENT" and dt >= win5))
    if term:
        f["sup2_terminal_any_hist"] = 1.0
        f["sup2_terminal_n_5y"] = float(sum(1 for dt, _ in term if dt >= win5))
    return f


def _days_between(d1: int, d2: int) -> int:
    """YYYYMMDD int 두 개의 근사 일수 차(d2-d1)."""
    def to_ord(d):
        y, m, day = d // 10000, (d // 100) % 100, d % 100
        return y * 365 + m * 30 + day
    return max(0, to_ord(d2) - to_ord(d1))


def build() -> pd.DataFrame:
    s2c = build_stock_to_corp()
    base = PROJECT_ROOT / "preprocess" / "data" / "processed_fixed_v1" / "fixed_N1" / "exp-A"
    parts = []
    for sp in ["train", "valid", "test"]:
        df = pd.read_csv(base / f"{sp}.csv", dtype={"stock_code": str})
        parts.append(df[["stock_code", "year", "quarter"]])
    keys = pd.concat(parts).drop_duplicates().reset_index(drop=True)
    keys["stock_code"] = keys["stock_code"].str.zfill(6)
    print(f"[build v2] {len(keys)} keys, {keys['stock_code'].nunique()} stocks")

    # corp별 이벤트 캐시
    ev_cache: dict[str, list] = {}
    rows = []
    for _, r in keys.iterrows():
        s = r["stock_code"]; y = int(r["year"]); q = r["quarter"]
        corp = s2c.get(s)
        if corp not in ev_cache:
            ev_cache[corp] = _load_corp_events(corp) if corp else []
        cutoff = y * 10000 + QUARTER_END_MMDD.get(q, 1231)
        feats = _features_at(ev_cache[corp], cutoff)
        rows.append({"stock_code": s, "year": y, "quarter": q, **feats})
    out = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_DIR / "supervision_features_v2.csv", index=False)
    print(f"[saved] {(OUT_DIR / 'supervision_features_v2.csv').relative_to(PROJECT_ROOT)} ({len(out)} rows)")

    # 진단: 라벨 교차 (train)
    labels = pd.read_csv(base / "train.csv", dtype={"stock_code": str})[["stock_code", "year", "quarter", "label"]]
    labels["stock_code"] = labels["stock_code"].str.zfill(6)
    m = out.merge(labels, on=["stock_code", "year", "quarter"], how="inner")
    print("\n[train 라벨 vs sup2_early_any_hist]")
    print(m.groupby(["label", "sup2_early_any_hist"]).size().unstack(fill_value=0).to_string())
    print("\n[train 라벨 vs sup2_terminal_any_hist]")
    print(m.groupby(["label", "sup2_terminal_any_hist"]).size().unstack(fill_value=0).to_string())
    for c in ALL_FEATURES:
        nz = (out[c] != (9999.0 if c == "sup2_days_since_early" else 0.0)).mean()
        print(f"  {c:<26} nonzero/obs ratio = {nz:.3f}")
    return out


if __name__ == "__main__":
    build()
