# Raw JSON 스키마 분석

분석 대상: S3 raw JSON 샘플 2건 (healthy 1건, delisted 1건)

| 구분 | 종목코드 | 기업명 | 섹터 | 연도 | 분기 |
|---|---|---|---|---|---|
| healthy | 001810 | (확인 필요) | Materials | 2025 | Q1, H1, Q3, ANNUAL |
| delisted | 024810 | 이화전기 | Industrials | 2024 | Q1, H1, Q3, ANNUAL |

---

## 1. JSON 최상위 구조

파일 1개 = JSON **배열** (리스트). 각 원소가 하나의 계정과목 행.

```json
[
  { "rcept_no": "...", "sj_div": "BS", "account_id": "ifrs-full_Assets", ... },
  { "rcept_no": "...", "sj_div": "CIS", "account_id": "ifrs-full_Revenue", ... },
  ...
]
```

항목 수는 기업/분기에 따라 다름 (147~231개).

---

## 2. 키 목록 및 타입

### 공통 키 (모든 파일)

| 키 | 타입 | 설명 | 예시 |
|---|---|---|---|
| `rcept_no` | str | 접수번호 | `"20250321001760"` |
| `reprt_code` | str | 보고서 코드 | `"11011"` |
| `bsns_year` | str | 사업연도 | `"2024"` |
| `corp_code` | str | DART 고유 기업코드 | `"00145738"` |
| `sj_div` | str | 재무제표 구분 | `"BS"`, `"CIS"`, `"CF"`, `"SCE"` |
| `sj_nm` | str | 재무제표명 (한글) | `"재무상태표"` |
| `account_id` | str | IFRS/DART 계정 ID | `"ifrs-full_Assets"` |
| `account_nm` | str | 계정명 (한글) | `"자산총계"` |
| `account_detail` | str | 세부 구분 | `"-"` 또는 SCE 멤버 경로 |
| `thstrm_nm` | str | 당기 명칭 | `"제 60 기"` |
| `thstrm_amount` | str | **당기 금액** | `"279888900614"` |
| `frmtrm_nm` | str | 전기 명칭 | `"제 59 기"` |
| `frmtrm_amount` | str | **전기 금액** | `"303176861845"` |
| `ord` | str | 표시 순서 | `"7"` |
| `currency` | str | 통화 | `"KRW"` (전체 KRW 고정) |

### ANNUAL 전용 키

| 키 | 타입 | 설명 |
|---|---|---|
| `bfefrmtrm_nm` | str | 전전기 명칭 |
| `bfefrmtrm_amount` | str | 전전기 금액 |

### 분기(Q1/H1/Q3) CIS/CF 전용 키

| 키 | 타입 | 설명 |
|---|---|---|
| `thstrm_add_amount` | str | 당기 누적 금액 (추가) |
| `frmtrm_q_amount` | str | 전기 동분기 금액 |
| `frmtrm_q_nm` | str | 전기 동분기 명칭 |
| `frmtrm_add_amount` | str | 전기 누적 금액 (추가) |

---

## 3. `reprt_code` 매핑

| reprt_code | 분기 |
|---|---|
| `11013` | Q1 (1분기) |
| `11012` | H1 (반기) |
| `11014` | Q3 (3분기) |
| `11011` | ANNUAL (사업보고서) |

---

## 4. `sj_div` 재무제표 구분

| sj_div | sj_nm | 설명 | 고유 계정 수 |
|---|---|---|---|
| `BS` | 재무상태표 | Balance Sheet | 113 |
| `CIS` | 포괄손익계산서 | Comprehensive Income Statement | 54 |
| `CF` | 현금흐름표 | Cash Flow Statement | 83 |
| `SCE` | 자본변동표 | Statement of Changes in Equity | 29 |

---

## 5. `account_detail` 값

- 대부분 `"-"` (단일 값, 주 재무제표 항목)
- SCE 항목에서 멤버 경로 존재:
  - `"연결재무제표 [member]"`
  - `"자본 [구성요소]|지배기업의 소유주에게 귀속되는 지분 [구성요소]|이익잉여금 [구성요소]"`
  - 등 (파이프 `|` 로 계층 구분)

**비율 계산 시**: `account_detail == "-"` 인 행만 사용

---

## 6. 금액 필드 특성

### 타입
- **모든 금액은 문자열(str)** — `"279888900614"`, `"0"`, `""`
- 숫자 파싱 필요: `int()` 또는 `float()` 변환

### 결측/이상 표현

| 표현 | 빈도 | 발생 위치 |
|---|---|---|
| `""` (빈 문자열) | 보통 | `thstrm_amount`, `bfefrmtrm_amount` 등 |
| 키 자체 누락 | 분기 파일 | `bfefrmtrm_*` 키 (ANNUAL에만 존재) |
| `"0"` | 자주 | 해당 항목이 0원인 경우 |
| `null` / `None` | 미발견 | — |
| `"-"` | 미발견 | 금액에서는 미사용 (`account_detail`에만 사용) |

### 분기별 금액 키 가용성

| 키 | ANNUAL | Q1/H1/Q3 |
|---|---|---|
| `thstrm_amount` | O | O |
| `frmtrm_amount` | O (전기 ANNUAL) | O (전기말 BS) / 없음 (CIS) |
| `bfefrmtrm_amount` | O (전전기) | X |
| `thstrm_add_amount` | X | O (CIS/CF 누적) |
| `frmtrm_q_amount` | X | O (전기 동분기) |

**핵심 차이**: 분기 CIS에서 `frmtrm_amount`가 없음 → 매출액/순이익/영업이익 YoY 증가율 계산 불가 (ANNUAL만 가능)

---

## 7. 주요 계정 ID 매핑 (비율 계산용)

### BS (재무상태표)

| 계정 | account_id | 비고 |
|---|---|---|
| 자산총계 | `ifrs-full_Assets` | |
| 유동자산 | `ifrs-full_CurrentAssets` | |
| 비유동자산 | `ifrs-full_NoncurrentAssets` | |
| 자본총계 | `ifrs-full_Equity` | |
| 부채총계 | `ifrs-full_Liabilities` | |
| 유동부채 | `ifrs-full_CurrentLiabilities` | |
| 비유동부채 | `ifrs-full_NoncurrentLiabilities` | |
| 매출채권 | `ifrs-full_CurrentTradeReceivables` 또는 `dart_ShortTermTradeReceivable` | 기업마다 다름 |
| 재고자산 | `ifrs-full_Inventories` | |
| 현금및현금성자산 | `ifrs-full_CashAndCashEquivalents` | |
| 유형자산 | `ifrs-full_PropertyPlantAndEquipment` | |
| 무형자산 | `ifrs-full_IntangibleAssetsAndGoodwill` 또는 `ifrs-full_IntangibleAssetsOtherThanGoodwill` | 기업마다 다름 |
| 단기차입금 | `ifrs-full_ShorttermBorrowings` | |
| 유동성장기차입금 | `ifrs-full_CurrentPortionOfLongtermBorrowings` | |
| 장기차입금 | `ifrs-full_LongtermBorrowings` | 없는 기업 있음 |
| 자본금 | `ifrs-full_IssuedCapital` | |
| 이익잉여금 | `ifrs-full_RetainedEarnings` | |

### CIS (포괄손익계산서)

| 계정 | account_id |
|---|---|
| 매출액 | `ifrs-full_Revenue` |
| 매출원가 | `ifrs-full_CostOfSales` |
| 매출총이익 | `ifrs-full_GrossProfit` |
| 영업이익 | `dart_OperatingIncomeLoss` |
| 당기순이익 | `ifrs-full_ProfitLoss` |

---

## 8. Output CSV 비율 계산식

기존 output CSV 샘플과의 역검증으로 확정한 계산식:

### 성장성

| 비율 | 계산식 | ANNUAL | 분기 |
|---|---|---|---|
| 총자산증가율 | (자산총계_당기 - 자산총계_전기) / \|자산총계_전기\| × 100 | O | O |
| 유동자산증가율 | (유동자산_당기 - 유동자산_전기) / \|유동자산_전기\| × 100 | O | O |
| 매출액증가율 | (매출액_당기 - 매출액_전기) / \|매출액_전기\| × 100 | O | X (frmtrm 없음) |
| 순이익증가율 | (순이익_당기 - 순이익_전기) / \|순이익_전기\| × 100 | O | X |
| 영업이익증가율 | (영업이익_당기 - 영업이익_전기) / \|영업이익_전기\| × 100 | O | X |

### 수익성

| 비율 | 계산식 |
|---|---|
| 매출액순이익률 | 당기순이익 / 매출액 × 100 |
| 매출총이익률 | 매출총이익 / 매출액 × 100 |
| 자기자본순이익률 (ROE) | 당기순이익 / 자본총계 × 100 |
| 매출원가율 | 매출원가 / 매출액 × 100 |

### 활동성 (회전율)

| 비율 | 계산식 | 비고 |
|---|---|---|
| 매출채권회전율 | 매출액 / 매출채권 | 기말 잔액 사용 (평균 아님) |
| 재고자산회전율 | **매출원가** / 재고자산 | 매출액이 아닌 매출원가 사용 |
| 총자본회전율 | 매출액 / 자산총계 | 기말 잔액 |
| 유형자산회전율 | 매출액 / 자산총계 | **현재 총자본회전율과 동일값** (확인 필요) |

### 안정성

| 비율 | 계산식 | 비고 |
|---|---|---|
| 부채비율 | **자산총계 / 자본총계** × 100 | 일반적 부채비율(부채/자본)이 아닌 재무레버리지 비율 |
| 유동비율 | 유동자산 / 유동부채 × 100 | |
| 자기자본비율 | 자본총계 / 자산총계 × 100 | |
| 당좌비율 | (유동자산 - 재고자산) / 유동부채 × 100 | |
| 순운전자본비율 | (유동자산 - 유동부채) / 자산총계 × 100 | |
| 차입금의존도 | (단기차입금 + 유동성장기차입금 + 장기차입금) / 자산총계 × 100 | |
| 현금비율 | 현금및현금성자산 / 유동부채 × 100 | |

### 자본효율

| 비율 | 계산식 |
|---|---|
| 총자본영업이익률 | 영업이익 / 자산총계 × 100 |
| 총자본순이익률 | 당기순이익 / 자산총계 × 100 |
| 유보액/납입자본비율 | 이익잉여금 / 자본금 × 100 |
| 총자본투자효율 | 당기순이익 / 자산총계 (소수, 비율 아님) |

### 원시값 (비율 아님)

| 컬럼 | 출처 |
|---|---|
| 유형자산 | BS `ifrs-full_PropertyPlantAndEquipment` thstrm_amount |
| 무형자산 | BS `ifrs-full_IntangibleAssetsAndGoodwill` 또는 `IntangibleAssetsOtherThanGoodwill` |
| 무형자산상각비 | 현재 빈 값 (CF/주석에서 추출 필요) |
| 유형자산상각비 | 현재 빈 값 |
| 감가상각비 | 현재 빈 값 |
| 비유동자산장기적합률 | 현재 빈 값 |

---

## 9. healthy vs delisted 구조 차이

| 항목 | healthy (001810) | delisted (024810) |
|---|---|---|
| 항목 수 (ANNUAL) | 175 | 231 |
| 매출채권 account_id | `ifrs-full_CurrentTradeReceivables` | `dart_ShortTermTradeReceivable` |
| 무형자산 account_id | `ifrs-full_IntangibleAssetsAndGoodwill` | `ifrs-full_IntangibleAssetsOtherThanGoodwill` |
| 장기차입금 | `ifrs-full_LongtermBorrowings` 존재 | 존재하지 않음 |
| 납입자본 항목 | `-표준계정코드 미사용-` (별도 행) | 없음 (자본금+주식발행초과금 합산) |
| 분기 CIS frmtrm | 없음 | 없음 |

**핵심**: 계정 ID가 기업마다 다를 수 있으므로 fallback 매핑 필요.

---

## 10. 확인 필요 사항

1. **유형자산회전율**: 현재 총자본회전율과 동일값 — 의도된 것인지 버그인지 확인 필요
2. **부채비율 정의**: 자산/자본 (재무레버리지)로 구현되어 있음. 일반적 부채비율(부채/자본)과 다름
3. **감가상각비 계열**: 3개 컬럼 모두 빈 값 — CF에서 추출하는 로직 추가 필요 여부
4. **비유동자산장기적합률**: 모든 분기 빈 값 — 계산 조건 확인 필요
5. **기업명(corp_name)**: raw JSON에 포함되지 않음 — 별도 매핑 테이블 필요
