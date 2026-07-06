"""DART 계정과목명 → 표준 키 매핑.

OpenDART에서 반환하는 account_nm은 기업마다 표현이 다를 수 있다.
이 모듈은 다양한 변형을 표준 키로 통합한다.

사용 가능한 표준 키(Standard Key) 목록
────────────────────────────────────────
■ 재무상태표 (BS)
  total_assets           자산총계
  current_assets         유동자산
  non_current_assets     비유동자산
  tangible_assets        유형자산
  intangible_assets      무형자산
  trade_receivables      매출채권
  inventories            재고자산
  cash                   현금및현금성자산
  total_liabilities      부채총계
  current_liabilities    유동부채
  short_term_borrowings  단기차입금
  long_term_borrowings   장기차입금
  bonds_payable          사채
  total_equity           자본총계 (= 자기자본)
  paid_in_capital        납입자본금 (= 자본금)
  retained_earnings      이익잉여금
  capital_surplus        자본잉여금

■ 손익계산서 (IS)
  revenue                매출액
  cost_of_sales          매출원가
  gross_profit           매출총이익
  operating_income       영업이익(손실)
  net_income             당기순이익(손실)
  interest_expense       이자비용

■ 현금흐름표 (CF) – 감가상각비 관련
  depreciation           유형자산감가상각비
  amortization           무형자산상각비
"""

from __future__ import annotations

import re
from typing import Any

# ── 계정명 패턴 → 표준 키 ─────────────────────────────────────
# 각 튜플: (표준키, sj_div 필터 또는 None, 정규식 패턴)
# 매칭 순서가 중요: 먼저 정의된 패턴이 우선.
ACCOUNT_PATTERNS: list[tuple[str, str | None, str]] = [
    # ─── BS (재무상태표) ───
    ("total_assets",          "BS", r"자산\s*총계"),
    ("current_assets",        "BS", r"유동\s*자산$"),
    ("non_current_assets",    "BS", r"비유동\s*자산$"),
    ("tangible_assets",       "BS", r"유형\s*자산$"),
    ("intangible_assets",     "BS", r"무형\s*자산$|영업권\s*이외의\s*무형자산"),
    ("trade_receivables",     "BS", r"매출\s*채권|단기매출채권"),
    ("inventories",           "BS", r"재고\s*자산$"),
    ("cash",                  "BS", r"현금\s*(및|과)\s*현금\s*성?\s*자산"),
    ("total_liabilities",     "BS", r"부채\s*총계"),
    ("current_liabilities",   "BS", r"유동\s*부채$"),
    ("short_term_borrowings", "BS", r"단기\s*차입금"),
    ("long_term_borrowings",  "BS", r"장기\s*차입금"),
    ("bonds_payable",         "BS", r"^사채$"),
    ("total_equity",          "BS", r"자본\s*총계"),
    ("paid_in_capital",       "BS", r"^자본금$|납입\s*자본"),
    ("retained_earnings",     "BS", r"이익\s*잉여금"),
    ("capital_surplus",       "BS", r"자본\s*잉여금"),

    # ─── IS (손익계산서) ───

    # [2025-04-17] revenue: 'I. 매출액', 'Ⅰ.매출액' 등 로마숫자 prefix 대응 → ^ 제거.
    # 단, '매출원가' 등과의 오매핑 방지를 위해 $ 앵커는 유지.
    # '상품매출액', '제품매출액', '순매출액' 등 매출 유형 prefix 변형 추가.
    # '수익(매출액)' 표현 추가 (일부 기업이 IS 없이 CIS에만 이 형태로 공시).
    # [2025-04-17] revenue 2차: '영업수익(매출액)' 완전 매칭 추가.
    ("revenue",               "IS",  r"매출액$|^매출$|^수익\s*\(매출액\)$|^영업\s*수익$|^수익$"
                                     r"|상품\s*매출액$|제품\s*매출액$|순\s*매출액$"
                                     r"|수익\s*\(매출액\)|영업\s*수익\s*\(매출액\)"),
    ("cost_of_sales",         "IS",  r"매출\s*원가"),
    ("gross_profit",          "IS",  r"매출\s*총이익|매출\s*총\s*손익"),

    # [2025-04-17] operating_income: 손실 기업이 '영업이익(손실)' 대신
    # '영업손실' 단독으로 공시하는 경우 추가.
    # [2025-04-17] operating_income 2차: '영업순손익' 추가.
    ("operating_income",      "IS",  r"영업\s*이익|영업\s*손익|^영업손실$|영업\s*순손익"),

    # [2025-04-17] net_income: 분기/반기 보고서에서 보고 기간명을 앞에 붙여 공시.
    # '분기순이익(손실)', '반기순이익(손실)', '분기순손실', '반기순손실' 등 추가.
    # '연결분기순이익(손실)', '당분기순이익(손실)', '계속영업분기순이익(손실)' 추가.
    # '계속영업이익(손실)'은 중단영업이 없는 경우 실질적 순이익이므로 포함.
    # 단, '법인세비용차감전분기순이익'은 세전이익이므로 의도적으로 제외.
    # [2025-04-17] net_income 2차: '당기순손익', '당기순손실' 단독 표현 추가.
    ("net_income",            "IS",  r"당기\s*순이익|당기순이익|당기\s*순\s*손익|당기\s*순손실"
                                     r"|당기\s*순손익"
                                     r"|분기\s*순이익|분기\s*순손실|분기\s*순\s*손익"
                                     r"|반기\s*순이익|반기\s*순손실|반기\s*순\s*손익"
                                     r"|연결\s*분기\s*순이익|연결\s*반기\s*순이익"
                                     r"|당\s*분기\s*순이익|당\s*반기\s*순이익"
                                     r"|계속\s*영업\s*분기\s*순이익|계속\s*영업\s*반기\s*순이익"
                                     r"|^계속영업이익$|^계속영업손실$"),
    ("interest_expense",      "IS",  r"이자\s*비용"),

    # ─── CIS (포괄손익계산서) ───
    # IS에서 매핑 실패 시 CIS에서 fallback 매핑.
    # IS가 matched_keys에 먼저 등록되면 CIS는 자동 스킵됨.
    # IS와 동일한 수정 내용 적용. [2025-04-17]

    # [2025-04-17] revenue 2차: IS와 동일하게 '영업수익(매출액)' 완전 매칭 추가.
    ("revenue",               "CIS", r"매출액$|^매출$|^수익\s*\(매출액\)$|^영업\s*수익$|^수익$"
                                     r"|상품\s*매출액$|제품\s*매출액$|순\s*매출액$"
                                     r"|수익\s*\(매출액\)|영업\s*수익\s*\(매출액\)"),
    ("cost_of_sales",         "CIS", r"매출\s*원가"),
    ("gross_profit",          "CIS", r"매출\s*총이익|매출\s*총\s*손익"),

    # [2025-04-17] operating_income 2차: IS와 동일하게 '영업순손익' 추가.
    ("operating_income",      "CIS", r"영업\s*이익|영업\s*손익|^영업손실$|영업\s*순손익"),

    # [2025-04-17] net_income 2차: IS와 동일하게 '당기순손익' 추가.
    ("net_income",            "CIS", r"당기\s*순이익|당기순이익|당기\s*순\s*손익|당기\s*순손실"
                                     r"|당기\s*순손익"
                                     r"|분기\s*순이익|분기\s*순손실|분기\s*순\s*손익"
                                     r"|반기\s*순이익|반기\s*순손실|반기\s*순\s*손익"
                                     r"|연결\s*분기\s*순이익|연결\s*반기\s*순이익"
                                     r"|당\s*분기\s*순이익|당\s*반기\s*순이익"
                                     r"|계속\s*영업\s*분기\s*순이익|계속\s*영업\s*반기\s*순이익"
                                     r"|^계속영업이익$|^계속영업손실$"),
    ("interest_expense",      "CIS", r"이자\s*비용"),

    # ─── CF (현금흐름표) – 감가상각비 ───
    ("depreciation",          "CF", r"유형\s*자산\s*감가\s*상각비|감가\s*상각비"),
    ("amortization",          "CF", r"무형\s*자산\s*상각비|무형자산상각비"),
]


def _parse_amount(raw: Any) -> float | None:
    """DART 금액 문자열 → float. 파싱 실패 시 None."""
    if raw is None:
        return None
    s = str(raw).strip().replace(",", "").replace(" ", "")
    if not s or s == "-":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def extract_standard_items(
    dart_items: list[dict[str, Any]],
) -> dict[str, dict[str, float | None]]:
    """
    DART 재무제표 항목 리스트로부터 표준 키별 금액을 추출.

    Returns:
        {
          "standard_key": {
            "thstrm": 당기 금액,
            "frmtrm": 전기 금액,
            "bfefrmtrm": 전전기 금액,
          },
          ...
        }
    """
    result: dict[str, dict[str, float | None]] = {}
    # 이미 매핑된 키는 중복 방지 (먼저 매칭된 것이 우선)
    matched_keys: set[str] = set()

    compiled = [
        (key, sj_div, re.compile(pattern))
        for key, sj_div, pattern in ACCOUNT_PATTERNS
    ]

    # [2025-04-17] 중단영업/계속영업 분리 구조 대응용 임시 저장소.
    # 일부 기업은 당기순이익 대신 '계속영업이익(손실)' + '중단영업이익(손실)'으로 분리 공시.
    # 이 경우 두 값을 합산해야 실제 당기순이익이 됨.
    # 상폐 직전 기업일수록 중단영업손실이 크게 나타나는 경향이 있어 반드시 합산 필요.
    _CONTINUING_RE   = re.compile(r"계속\s*영업\s*이익|계속\s*영업\s*손익|계속\s*영업\s*손실")
    _DISCONTINUED_RE = re.compile(r"중단\s*영업\s*이익|중단\s*영업\s*손익|중단\s*영업\s*손실")

    _periods = ("thstrm", "frmtrm", "bfefrmtrm")
    _continuing:   dict[str, float | None] | None = None
    _discontinued: dict[str, float | None] | None = None

    for item in dart_items:
        account_nm = (item.get("account_nm") or "").strip()
        sj_div = (item.get("sj_div") or "").strip()
        if not account_nm:
            continue

        # 계속영업/중단영업 항목 별도 수집 (IS 우선, 없으면 CIS)
        if sj_div in ("IS", "CIS"):
            if _continuing is None and _CONTINUING_RE.search(account_nm):
                _continuing = {
                    p: _parse_amount(item.get(f"{p}_amount")) for p in _periods
                }
            if _discontinued is None and _DISCONTINUED_RE.search(account_nm):
                _discontinued = {
                    p: _parse_amount(item.get(f"{p}_amount")) for p in _periods
                }

        for std_key, filter_sj, regex in compiled:
            if std_key in matched_keys:
                continue
            if filter_sj and sj_div != filter_sj:
                continue
            if regex.search(account_nm):
                result[std_key] = {
                    "thstrm": _parse_amount(item.get("thstrm_amount")),
                    "frmtrm": _parse_amount(item.get("frmtrm_amount")),
                    "bfefrmtrm": _parse_amount(item.get("bfefrmtrm_amount")),
                }
                matched_keys.add(std_key)
                break

    # [2025-04-17] net_income 미매핑 시 계속영업 + 중단영업 합산으로 보정.
    # 처리 우선순위:
    #   1) 패턴 매핑 성공 → 그대로 사용
    #   2) 계속영업 + 중단영업 둘 다 있음 → 합산
    #   3) 계속영업만 있음 → 계속영업 값만 사용
    #   4) 둘 다 없음 → NaN (처리 불가)
    if "net_income" not in matched_keys and _continuing is not None:
        def _add(a: float | None, b: float | None) -> float | None:
            if a is None and b is None:
                return None
            return (a or 0.0) + (b or 0.0)

        result["net_income"] = {
            p: _add(_continuing[p], _discontinued[p] if _discontinued else None)
            for p in _periods
        }
        matched_keys.add("net_income")

    return result