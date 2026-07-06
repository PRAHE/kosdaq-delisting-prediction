# Data Contract For A

## 목적

A가 재무비율 산출 결과를 기업별 파일로 분리하지 않고 병합·정제된 형식의 분석용 데이터셋(`clean_data`)으로 생성할 때 따라야 하는 스키마와 정규화 규칙을 정의한다.

이 문서는 현재 구현된 비율 산출 로직([preprocess/src/ratio_calculator.py](preprocess/src/ratio_calculator.py), [preprocess/src/account_mapper.py](preprocess/src/account_mapper.py))과 raw 구조 점검 문서([docs/raw_schema_check.md](docs/raw_schema_check.md))를 기준으로 작성한다.

## 산출물

- 파일 형식: CSV
- 인코딩: UTF-8
- 최종 산출물:
  - `clean_data_no_macro.csv` (재무비율만)
  - `clean_data.csv` (재무비율 + 거시지표)
- 권장 저장 위치:
  - `preprocess/data/output/clean_data_no_macro.csv`
  - `preprocess/data/output/clean_data.csv`

## 데이터 단위

- 1행 = 1개 기업의 1개 연도-분기 레코드
- 유일키: `(stock_code, year, quarter)`
- 허용 `quarter` 값: `Q1`, `H1`, `Q3`, `ANNUAL`

동일 기업/연도에 대해 최대 4개 행이 생성될 수 있으며, 분기 파일이 없으면 해당 분기 행은 생략될 수 있다.

## 입력 소스

`clean_data_no_macro.csv`는 재무비율 산출 결과를 기업/연도/분기 단위로 병합·정제해 생성한다.

1. DART raw JSON 기반 재무비율 산출 결과
   - 기업별 재무비율 CSV를 중간 산출물로 고정하지 않는다.
   - 현재 컬럼: `stock_code`, `corp_name`, `year`, `quarter`, `label` + 재무비율/재무항목 컬럼
2. 기업 메타데이터
   - 최소 필요 컬럼: `stock_code`, `gics_sector`
   - 선택 컬럼: `corp_name`
3. 거시지표 데이터
   - `merged_dataset_with_macro.csv` 생성 시 `merged_dataset.csv`에 조인한다.

재무비율 산출 결과에 `gics_sector`가 포함되어 있지 않으면 병합 단계에서 별도 메타데이터를 통해 반드시 보강해야 한다.

## 필수 컬럼

### 메타 컬럼

| 컬럼명 | 필수 여부 | dtype | 규칙 |
|---|---|---|---|
| `stock_code` | 필수 | str | 6자리 zero-padded 문자열 |
| `year` | 필수 | int | 사업연도 |
| `quarter` | 필수 | str | `Q1`, `H1`, `Q3`, `ANNUAL` 중 하나 |
| `label` | 필수 | int | `0`=healthy, `1`=delisted |
| `gics_sector` | 필수 | str | GICS sector 명칭 |
| `corp_name` | 선택 | str | 분석에는 필수 아님. 있으면 유지 권장 |

### 재무비율/재무항목 컬럼

아래 컬럼은 현재 비율 산출기 출력 스키마와 동일해야 한다.

| 컬럼명 | dtype |
|---|---|
| `총자산증가율` | float |
| `유동자산증가율` | float |
| `매출액순이익률` | float |
| `매출총이익률` | float |
| `자기자본순이익률` | float |
| `매출채권회전율` | float |
| `재고자산회전율` | float |
| `총자본회전율` | float |
| `유형자산회전율` | float |
| `매출원가율` | float |
| `부채비율` | float |
| `유동비율` | float |
| `자기자본비율` | float |
| `당좌비율` | float |
| `비유동자산장기적합률` | float |
| `순운전자본비율` | float |
| `차입금의존도` | float |
| `현금비율` | float |
| `유형자산` | float |
| `무형자산` | float |
| `총자본영업이익률` | float |
| `총자본순이익률` | float |
| `유보액/납입자본비율` | float |
| `총자본투자효율` | float |

#### 전처리 단계에서 제거된 컬럼 (결측률 50% 초과)

| 컬럼명 | 결측률 | 제거 사유 |
|---|---|---|
| `매출액증가율` | 68.3% | 전기 데이터 없는 첫 수집 연도의 구조적 결측 |
| `순이익증가율` | 68.3% | 전기 데이터 없는 첫 수집 연도의 구조적 결측 |
| `영업이익증가율` | 67.7% | 전기 데이터 없는 첫 수집 연도의 구조적 결측 |
| `무형자산상각비` | 76.6% | CF 계정과목 매핑 미수집 |
| `유형자산상각비` | 74.3% | CF 계정과목 매핑 미수집 |
| `감가상각비` | 74.1% | CF 계정과목 매핑 미수집 |

### 거시지표 컬럼 (`clean_data.csv` 전용)

| 컬럼명 | dtype | 설명 |
|---|---|---|
| `credit_spread` | float | 신용 스프레드 |
| `kosdaq_return` | float | 코스닥 수익률 |
| `gdp_growth_yoy` | float | GDP 전년동기비 성장률 |
| `usdkrw_chg` | float | 원달러 환율 변동률 |
| `vix_avg` | float | VIX 평균 |
| `cpi_yoy` | float | 소비자물가 전년동기비 |

## 정규화 규칙

### 1. 키/메타 규칙

- `stock_code`는 반드시 문자열로 저장하고, 앞자리 `0`을 보존한다.
- `year`는 정수형으로 저장한다.
- `quarter`는 원본 DART 기준값을 그대로 사용한다.
- `label`은 정수형 `0/1`로 통일한다.
- `gics_sector`는 빈 값이 허용되지 않는다.

### 2. 결측치 규칙

- 결측치는 모두 `NaN`으로 통일한다.
- 빈 문자열 `""`, 문자열 `"None"`, 문자열 `"null"`, 하이픈 `"-"`은 결측으로 정규화한다.
- 숫자형 컬럼에 문자열이 남아 있으면 안 된다.

### 3. 중복 규칙

- `(stock_code, year, quarter)` 기준 중복 행은 허용하지 않는다.
- 중복 발생 시 단순 keep-first 하지 말고 원인 확인 후 제거해야 한다.

### 4. 컬럼명 규칙

- 컬럼명은 현재 비율 산출기 출력명을 그대로 유지한다.
- 병합 단계에서 임의 축약, 번역, snake_case 변환을 하지 않는다.

## 병합 규칙

1. 재무비율 산출 결과를 기업/연도/분기 단위 행으로 구성한다.
2. 각 행에 `gics_sector`를 메타데이터에서 조인한다.
3. `stock_code`, `year`, `quarter` 기준으로 중복 여부를 검증한다.
4. 숫자형 컬럼을 일괄 float 변환한다.
5. 결측치를 `NaN`으로 통일한다.
6. `clean_data.csv`는 `clean_data_no_macro.csv`에 거시지표 컬럼을 추가 조인해 생성한다.
7. 최종 컬럼 순서는 아래 순서를 권장한다.

```text
stock_code, corp_name, year, quarter, label, gics_sector, [재무비율/재무항목 컬럼...]
```

`corp_name`은 유지 권장이지만, 분석 파이프라인 최소 요건은 `stock_code`, `year`, `quarter`, `label`, `gics_sector`와 수치 컬럼들이다.

## 검증 체크리스트

최종 `clean_data_no_macro.csv`는 아래 조건을 모두 만족해야 한다.

- 필수 컬럼 존재
- `(stock_code, year, quarter)` 중복 없음
- `stock_code` 6자리 문자열 유지
- `quarter` 값이 `Q1`, `H1`, `Q3`, `ANNUAL` 외 값을 포함하지 않음
- `label` 값이 `0`, `1` 외 값을 포함하지 않음
- `gics_sector` 결측 없음
- 재무비율/재무항목 컬럼이 모두 float 또는 결측으로만 구성됨
- 빈 문자열 기반 결측이 남아 있지 않음

## 비율 계산식 기준

병합 단계에서는 원칙적으로 재무비율을 재계산하지 않고, 재무비율 산출 단계에서 계산된 값을 그대로 적재한다.

다만 검증이 필요할 경우 계산식 기준은 아래 문서를 따른다.

- raw 구조 및 계정 매핑: [docs/raw_schema_check.md](docs/raw_schema_check.md)
- 계정 매핑: [preprocess/src/account_mapper.py](preprocess/src/account_mapper.py)
- 비율 산출 로직: [preprocess/src/ratio_calculator.py](preprocess/src/ratio_calculator.py)
- ETL 파이프라인: [preprocess/src/etl.py](preprocess/src/etl.py)

현재 확인된 구현상 주의사항은 다음과 같다.

- `감가상각비`, `무형자산상각비`, `유형자산상각비`는 CF(현금흐름표)에서 추출하며, 결측률 50% 초과로 전처리 단계에서 제거되었다.
- `매출액증가율`, `순이익증가율`, `영업이익증가율`은 전기 데이터가 없는 첫 수집 연도의 구조적 결측으로 제거되었다.

## A에게 전달할 요청사항

- `gics_sector`를 안정적으로 조인할 수 있는 기준 메타데이터를 함께 관리할 것
- `stock_code`의 문자열 형식을 끝까지 유지할 것
- 병합 결과를 제출할 때 중복 검증 결과와 결측 정규화 여부를 함께 확인할 것
- 비율 재계산이 필요하면 현재 구현 기준과 불일치하는 컬럼이 없는지 사전 합의할 것
