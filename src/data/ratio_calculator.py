"""30개 재무비율 계산기.

이미지에 정의된 재무비율을 표준 키 기반 재무 항목으로부터 계산한다.

카테고리별 비율 목록
──────────────────
[성장성]  5개   총자산증가율 ~ 영업이익증가율
[수익성]  3개   매출액순이익률 ~ 자기자본순이익률
[활동성]  5개   매출채권회전율 ~ 매출원가율
[안정성] 13개   부채비율 ~ 감가상각비  (유형/무형자산 값 포함)
[가치평가] 4개  총자본영업이익률 ~ 총자본투자효율
"""

from __future__ import annotations

from typing import Any

# 타입 별칭
Items = dict[str, dict[str, float | None]]  # account_mapper 결과


def _get(items: Items, key: str, period: str = "thstrm") -> float | None:
    """items에서 특정 키의 특정 기간 값을 꺼낸다."""
    entry = items.get(key)
    if entry is None:
        return None
    return entry.get(period)


def _safe_div(numerator: float | None, denominator: float | None) -> float | None:
    """안전 나눗셈. 0 나눗셈이나 None은 None 반환."""
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def _pct(numerator: float | None, denominator: float | None) -> float | None:
    """(numerator / denominator) * 100. 비율(%) 계산."""
    val = _safe_div(numerator, denominator)
    return val * 100 if val is not None else None


def _growth(current: float | None, previous: float | None) -> float | None:
    """증가율(%) = (당기 - 전기) / 전기 * 100."""
    if current is None or previous is None or previous == 0:
        return None
    return (current - previous) / previous * 100


# ═══════════════════════════════════════════════════════════════
# 개별 비율 계산 함수
# ═══════════════════════════════════════════════════════════════

# ── 성장성 ────────────────────────────────────────────────────
def 총자산증가율(items: Items) -> float | None:
    """(기말총자산 - 기초총자산) / 기초총자산 * 100.
    기말 = thstrm, 기초 = frmtrm (BS 항목의 전기 잔액 = 당기 기초)."""
    return _growth(_get(items, "total_assets", "thstrm"),
                   _get(items, "total_assets", "frmtrm"))


def 유동자산증가율(items: Items) -> float | None:
    return _growth(_get(items, "current_assets", "thstrm"),
                   _get(items, "current_assets", "frmtrm"))


# [2025-04-17] 매출액증가율 / 순이익증가율 / 영업이익증가율 제거.
# DART 분기/반기 보고서는 IS 항목의 frmtrm을 비워 공시하는 경우가 많아
# frmtrm 기반 계산 시 Q1/H1/Q3 결측률 92%+로 사실상 사용 불가.
# 전년 동기(YoY) 방식으로 교체 → build_master_dataset.py의
# _add_yoy_growth_cols()에서 전담 계산 후 combined_raw.csv에 저장.


# ── 수익성 ────────────────────────────────────────────────────
def 매출액순이익률(items: Items) -> float | None:
    """순이익 / 매출액 * 100."""
    return _pct(_get(items, "net_income"), _get(items, "revenue"))


def 매출총이익률(items: Items) -> float | None:
    """매출총이익 / 매출액 * 100."""
    return _pct(_get(items, "gross_profit"), _get(items, "revenue"))


def 자기자본순이익률(items: Items) -> float | None:
    """순이익 / 자기자본 * 100  (ROE)."""
    return _pct(_get(items, "net_income"), _get(items, "total_equity"))


# ── 활동성 ────────────────────────────────────────────────────
def 매출채권회전율(items: Items) -> float | None:
    """매출액 / 매출채권."""
    return _safe_div(_get(items, "revenue"), _get(items, "trade_receivables"))


def 재고자산회전율(items: Items) -> float | None:
    """매출원가 / 재고자산."""
    return _safe_div(_get(items, "cost_of_sales"), _get(items, "inventories"))


def 총자본회전율(items: Items) -> float | None:
    """매출액 / 총자본 (= 자산총계)."""
    return _safe_div(_get(items, "revenue"), _get(items, "total_assets"))


def 유형자산회전율(items: Items) -> float | None:
    """매출액 / 유형자산."""
    return _safe_div(_get(items, "revenue"), _get(items, "tangible_assets"))


def 매출원가율(items: Items) -> float | None:
    """매출원가 / 매출액 * 100."""
    return _pct(_get(items, "cost_of_sales"), _get(items, "revenue"))


# ── 안정성 ────────────────────────────────────────────────────
def 부채비율(items: Items) -> float | None:
    """부채 / 자기자본 * 100."""
    return _pct(_get(items, "total_liabilities"), _get(items, "total_equity"))


def 유동비율(items: Items) -> float | None:
    """유동자산 / 유동부채 * 100."""
    return _pct(_get(items, "current_assets"), _get(items, "current_liabilities"))


def 자기자본비율(items: Items) -> float | None:
    """자기자본 / 총자산 * 100."""
    return _pct(_get(items, "total_equity"), _get(items, "total_assets"))


def 당좌비율(items: Items) -> float | None:
    """당좌자산 / 유동부채 * 100.
    당좌자산 = 유동자산 - 재고자산."""
    ca = _get(items, "current_assets")
    inv = _get(items, "inventories") or 0
    cl = _get(items, "current_liabilities")
    if ca is None:
        return None
    quick = ca - inv
    return _pct(quick, cl)


def 비유동자산장기적합률(items: Items) -> float | None:
    """비유동자산 / 장기차입금."""
    return _safe_div(_get(items, "non_current_assets"),
                     _get(items, "long_term_borrowings"))


def 순운전자본비율(items: Items) -> float | None:
    """순운전자본 / 총자본 * 100.
    순운전자본 = 유동자산 - 유동부채, 총자본 = 자산총계."""
    ca = _get(items, "current_assets")
    cl = _get(items, "current_liabilities")
    ta = _get(items, "total_assets")
    if ca is None or cl is None:
        return None
    nwc = ca - cl
    return _pct(nwc, ta)


def 차입금의존도(items: Items) -> float | None:
    """(장기+단기차입금+사채) / 총자본 * 100."""
    stb = _get(items, "short_term_borrowings") or 0
    ltb = _get(items, "long_term_borrowings") or 0
    bonds = _get(items, "bonds_payable") or 0
    ta = _get(items, "total_assets")
    total_borrowing = stb + ltb + bonds
    return _pct(total_borrowing, ta)


def 현금비율(items: Items) -> float | None:
    """현금예금 / 유동부채 * 100."""
    return _pct(_get(items, "cash"), _get(items, "current_liabilities"))


def 유형자산_값(items: Items) -> float | None:
    """유형자산 절대값."""
    return _get(items, "tangible_assets")


def 무형자산_값(items: Items) -> float | None:
    """무형자산 절대값."""
    return _get(items, "intangible_assets")


def 무형자산상각비_값(items: Items) -> float | None:
    """무형자산상각비 (CF에서 추출)."""
    return _get(items, "amortization")


def 유형자산상각비_값(items: Items) -> float | None:
    """유형자산감가상각비 (CF에서 추출)."""
    return _get(items, "depreciation")


def 감가상각비(items: Items) -> float | None:
    """유형자산상각비 + 무형자산상각비."""
    dep = _get(items, "depreciation") or 0
    amo = _get(items, "amortization") or 0
    if _get(items, "depreciation") is None and _get(items, "amortization") is None:
        return None
    return dep + amo


# ── 가치평가 ──────────────────────────────────────────────────
def 총자본영업이익률(items: Items) -> float | None:
    """영업이익 / 총자본 * 100."""
    return _pct(_get(items, "operating_income"), _get(items, "total_assets"))


def 총자본순이익률(items: Items) -> float | None:
    """당기순이익 / 총자본 * 100."""
    return _pct(_get(items, "net_income"), _get(items, "total_assets"))


def 유보액_납입자본비율(items: Items) -> float | None:
    """유보액 / 납입자본금 * 100.
    유보액 ≈ 이익잉여금 + 자본잉여금."""
    re_ = _get(items, "retained_earnings") or 0
    cs = _get(items, "capital_surplus") or 0
    pic = _get(items, "paid_in_capital")
    if _get(items, "retained_earnings") is None:
        return None
    reserves = re_ + cs
    return _pct(reserves, pic)


def 총자본투자효율(items: Items) -> float | None:
    """(당기순이익 + 이자비용) / 총자본."""
    ni = _get(items, "net_income")
    ie = _get(items, "interest_expense") or 0
    ta = _get(items, "total_assets")
    if ni is None:
        return None
    return _safe_div(ni + ie, ta)


# ═══════════════════════════════════════════════════════════════
# 전체 비율 계산 오케스트레이션
# ═══════════════════════════════════════════════════════════════

# (카테고리, 비율명, 계산함수) 순서 리스트
RATIO_DEFINITIONS: list[tuple[str, str, Any]] = [
    # 성장성
    # 총자산/유동자산 증가율: BS 항목이라 frmtrm이 안정적으로 채워짐 → 기존 방식 유지
    ("성장성", "총자산증가율",       총자산증가율),
    ("성장성", "유동자산증가율",     유동자산증가율),
    # 매출액/순이익/영업이익 증가율: IS 항목이라 frmtrm 결측 92%+
    # → YoY 방식으로 교체, build_master_dataset.py에서 계산
    # 수익성
    ("수익성", "매출액순이익률",     매출액순이익률),
    ("수익성", "매출총이익률",       매출총이익률),
    ("수익성", "자기자본순이익률",   자기자본순이익률),
    # 활동성
    ("활동성", "매출채권회전율",     매출채권회전율),
    ("활동성", "재고자산회전율",     재고자산회전율),
    ("활동성", "총자본회전율",       총자본회전율),
    ("활동성", "유형자산회전율",     유형자산회전율),
    ("활동성", "매출원가율",         매출원가율),
    # 안정성
    ("안정성", "부채비율",           부채비율),
    ("안정성", "유동비율",           유동비율),
    ("안정성", "자기자본비율",       자기자본비율),
    ("안정성", "당좌비율",           당좌비율),
    ("안정성", "비유동자산장기적합률", 비유동자산장기적합률),
    ("안정성", "순운전자본비율",     순운전자본비율),
    ("안정성", "차입금의존도",       차입금의존도),
    ("안정성", "현금비율",           현금비율),
    ("안정성", "유형자산",           유형자산_값),
    ("안정성", "무형자산",           무형자산_값),
    ("안정성", "무형자산상각비",     무형자산상각비_값),
    ("안정성", "유형자산상각비",     유형자산상각비_값),
    ("안정성", "감가상각비",         감가상각비),
    # 가치평가
    ("가치평가", "총자본영업이익률", 총자본영업이익률),
    ("가치평가", "총자본순이익률",   총자본순이익률),
    ("가치평가", "유보액/납입자본비율", 유보액_납입자본비율),
    ("가치평가", "총자본투자효율",   총자본투자효율),
]

# CSV 컬럼 순서에 쓸 비율명 리스트
RATIO_NAMES: list[str] = [name for _, name, _ in RATIO_DEFINITIONS]


def compute_all_ratios(items: Items) -> dict[str, float | None]:
    """
    표준 키 기반 재무 항목으로부터 30개 비율을 모두 계산.

    Returns:
        {"총자산증가율": 12.34, "유동자산증가율": None, ...}
    """
    result: dict[str, float | None] = {}
    for _cat, name, func in RATIO_DEFINITIONS:
        try:
            result[name] = func(items)
        except Exception:
            result[name] = None
    return result