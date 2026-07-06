# B 담당자 작업 계획서

## 전제 조건

- `merged_dataset`은 아직 없음 (A 작업 완료 전)
- S3에 raw JSON만 존재 (`{healthy|delisted}/{gics_sector}/{ticker}_{year}_{quarter}.json`)
- 로컬에 `data/`, `src/`, `notebooks/`, `docs/` 디렉터리 없음 (신규 생성 필요)
- raw CSV 파일도 로컬에 없음 → S3에서 샘플 다운로드하거나 A에게 받아야 함

---

## 작업 순서 전체 흐름

```
Phase 1: 데이터 계약서 → Phase 2: 분석 함수 → Phase 3: EDA 노트북 → Phase 4: baseline 뼈대
```

Phase 1이 완료되어야 Phase 2~4에서 컬럼명/dtype을 확정적으로 쓸 수 있음.
다만 Phase 2~3은 컬럼 분류 로직을 **패턴 기반**으로 설계하면 Phase 1과 병행 가능.

---

## Phase 1: A에게 줄 데이터 계약서 작성

### 목표
A가 `merged_dataset`을 만들 때 지켜야 할 스펙을 문서로 확정

### 선행 작업
S3에서 raw JSON 샘플 2~3건 다운로드하여 구조 파악

```bash
# S3에서 샘플 확인 (정상 기업 1건, 상폐 기업 1건)
python -m s3.cli by-sector --status healthy --json
python -m s3.cli by-sector --status delisted --json
```

### 작업 단계

#### 1-1. S3 raw JSON 샘플 다운로드 및 구조 파악
- S3에서 healthy/delisted 각 1~2건 다운로드
- JSON 키 목록 전부 추출
- 어떤 필드가 메타(stock_code, year 등)이고 어떤 필드가 재무 수치인지 분류
- `label`, `gics_sector`가 JSON 안에 포함되는지 확인
- 결측 표현 방식 확인 (`null`, `""`, 누락 등)

```bash
# boto3로 직접 다운로드하거나 AWS CLI 사용
aws s3 cp s3://kw0ss-raw-data-s3/healthy/{sector}/{ticker}_{year}_{quarter}.json ./data/sample/
```

#### 1-2. raw 구조 정리 문서 작성
- **산출물**: `docs/raw_schema_check.md`
- 내용:
  - JSON 최상위 키 목록
  - 각 키의 값 타입 (str, int, float, list, null)
  - 메타 필드 vs 재무 항목 필드 구분표
  - quarter 값 종류 (Q1, H1, Q3, ANNUAL 등)
  - 결측/이상 표현 방식 목록

#### 1-3. 데이터 계약서 작성
- **산출물**: `docs/data_contract_for_A.md`
- 포함 내용:

| 항목 | 내용 |
|---|---|
| 파일 형식 | CSV (UTF-8) |
| 파일명 | `merged_dataset.csv` (단일 파일) |
| 필수 컬럼 | `stock_code`, `year`, `quarter`, `label`, `gics_sector` + 재무비율 컬럼들 |
| 유일키 | `(stock_code, year, quarter)` — 중복 불허 |
| dtype 규칙 | `stock_code`: str(6자리 zero-padded), `year`: int, `quarter`: str, `label`: int(0/1), 재무비율: float |
| 결측 표현 | `NaN` 통일 (빈 문자열, None 금지) |
| quarter 원본 보존 | DART 원본 값 그대로 (Q1, H1, Q3, ANNUAL) |
| ratio 계산식 | 주요 비율 계산식 명시 요청 (검증용) |

### 예상 소요
- 1-1: 30분 (S3 접근 + JSON 구조 파악)
- 1-2: 30분
- 1-3: 1시간

---

## Phase 2: 재사용 분석 함수 작성 (`src/analysis/utils.py`)

### 목표
EDA 노트북과 이후 분석에서 반복 사용할 함수들을 모듈로 분리

### 디렉터리 구조

```
src/
├── __init__.py
├── analysis/
│   ├── __init__.py
│   └── utils.py
└── baseline/
    ├── __init__.py
    └── run_baseline.py
```

### 작업 단계

#### 2-1. 데이터 로드 및 컬럼 분류 함수

```python
# 구현할 함수
def load_csv(path: str) -> pd.DataFrame
def classify_columns(df: pd.DataFrame) -> dict[str, list[str]]
    # 반환: {"meta": [...], "ratio": [...], "raw_financial": [...]}
def summarize_dataframe(df: pd.DataFrame) -> pd.DataFrame
    # shape, dtype 분포, 결측률 요약
```

- `classify_columns`는 컬럼명 패턴 기반으로 설계
  - meta: `stock_code`, `year`, `quarter`, `label`, `gics_sector`
  - 나머지 숫자형: ratio 또는 raw_financial로 분류
- Phase 1 결과로 패턴 확정 후 조정

#### 2-2. 결측 분석 함수

```python
def missing_summary(df: pd.DataFrame) -> pd.DataFrame
    # 컬럼별 결측 수, 결측률, dtype
def high_missing_columns(df: pd.DataFrame, threshold: float = 0.5) -> list[str]
def plot_missing_heatmap(df: pd.DataFrame) -> None
    # missingno 또는 seaborn 기반 시각화
```

#### 2-3. 단변량 분석 함수

```python
def plot_histogram(df: pd.DataFrame, feature: str, bins: int = 50) -> None
def plot_boxplot_by_label(df: pd.DataFrame, feature: str) -> None
def analyze_single_feature(df: pd.DataFrame, feature: str) -> dict
    # mean, median, std, min, max, IQR, skewness 등 요약 통계 반환
def compare_group_stats_by_label(df: pd.DataFrame, feature: str) -> pd.DataFrame
    # label=0 vs label=1 그룹 비교
```

#### 2-4. 상관관계 함수

```python
def plot_correlation_heatmap(df: pd.DataFrame, method: str = "pearson") -> None
def get_high_corr_pairs(df: pd.DataFrame, threshold: float = 0.9) -> list[tuple]
```

#### 2-5. 이상치 탐지

```python
def detect_outliers_iqr(df: pd.DataFrame, feature: str, factor: float = 1.5) -> pd.Series
    # bool Series 반환
```

### 의존 라이브러리
`requirements.txt`에 추가 필요:

```
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

### 예상 소요
- 2-1: 30분
- 2-2: 30분
- 2-3: 1시간
- 2-4: 30분
- 2-5: 20분

---

## Phase 3: EDA 템플릿 노트북 작성 (`notebooks/eda_template.ipynb`)

### 목표
`merged_dataset` 도착 시 즉시 실행할 수 있는 분석 노트북

### 의존
- Phase 2의 `src/analysis/utils.py` 함수를 import

### 노트북 섹션 구성

```
[1] 설정 & 데이터 로드
    - import, path 설정
    - load_csv()로 데이터 로드
    - df.shape, df.head() 확인

[2] 컬럼 분류
    - classify_columns()
    - 메타/비율/원천 컬럼 목록 출력

[3] 기본 통계
    - summarize_dataframe()
    - df.describe() 확장판

[4] 결측 분석
    - missing_summary()
    - high_missing_columns()
    - plot_missing_heatmap()

[5] 라벨 분포
    - label value_counts
    - 불균형 비율 확인
    - gics_sector별 label 분포

[6] 단변량 분포
    - 주요 비율 5~10개 선정
    - 각각 histogram + boxplot_by_label
    - analyze_single_feature()로 통계 요약

[7] 상관관계
    - plot_correlation_heatmap()
    - get_high_corr_pairs()
    - 다중공선성 후보 정리

[8] baseline 모델 (간단)
    - Phase 4에서 만든 함수 import
    - Logistic Regression + Decision Tree
    - 평가지표 출력
```

### 작업 방식
- 각 섹션은 마크다운 셀로 제목/설명 + 코드 셀로 구성
- 데이터 없이도 코드 구조가 보이도록 작성
- 샘플 데이터가 있으면 일부 셀만 테스트 실행

### 예상 소요
- 1.5~2시간

---

## Phase 4: baseline 모델 코드 뼈대 (`src/baseline/run_baseline.py`)

### 목표
`merged_dataset`이 오면 바로 돌릴 수 있는 baseline 분류 파이프라인

### 의존
- Phase 2의 `classify_columns()`, `load_csv()`

### 구현할 함수

#### 4-1. 전처리

```python
def prepare_features(df: pd.DataFrame, target: str = "label") -> tuple[pd.DataFrame, pd.Series]
    # 수치형 컬럼만 선택, 결측 대체(median), X/y 분리

def time_split(df, train_end: int, val_end: int) -> tuple[...]
    # year 기준: train <= train_end, val <= val_end, test = 나머지

def random_split(df, test_size: float = 0.2, val_size: float = 0.1, seed: int = 42) -> tuple[...]
```

#### 4-2. 모델 학습

```python
def train_logistic(X_train, y_train) -> LogisticRegression
def train_decision_tree(X_train, y_train) -> DecisionTreeClassifier
```

#### 4-3. 평가

```python
def evaluate(model, X_test, y_test) -> dict
    # 반환: {"f1", "precision", "recall", "roc_auc", "pr_auc"}

def print_report(results: dict[str, dict]) -> None
    # 모델별 지표 비교 테이블 출력
```

#### 4-4. 메인 실행

```python
def run_baseline(csv_path: str, split: str = "time") -> dict
    # 전체 파이프라인: 로드 → 전처리 → 분할 → 학습 → 평가

if __name__ == "__main__":
    run_baseline("data/merged_dataset.csv")
```

### 예상 소요
- 4-1: 30분
- 4-2: 20분
- 4-3: 30분
- 4-4: 20분

---

## 일정 요약

| 순서 | 작업 | 산출물 | 예상 소요 |
|---|---|---|---|
| 1 | S3 샘플 확인 + raw 구조 정리 | `docs/raw_schema_check.md` | 1시간 |
| 2 | 데이터 계약서 작성 | `docs/data_contract_for_A.md` | 1시간 |
| 3 | 분석 함수 작성 | `src/analysis/utils.py` | 2.5시간 |
| 4 | EDA 노트북 작성 | `notebooks/eda_template.ipynb` | 2시간 |
| 5 | baseline 뼈대 작성 | `src/baseline/run_baseline.py` | 1.5시간 |

**합계: 약 8시간 (넉넉히 1.5일)**

---

## 생성할 파일 목록

```
docs/
├── raw_schema_check.md          # Phase 1-2
└── data_contract_for_A.md       # Phase 1-3

src/
├── __init__.py
├── analysis/
│   ├── __init__.py
│   └── utils.py                 # Phase 2
└── baseline/
    ├── __init__.py
    └── run_baseline.py          # Phase 4

notebooks/
└── eda_template.ipynb           # Phase 3
```

---

## 추가 필요 패키지 (requirements.txt)

```
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

---

## A 결과물 도착 후 즉시 할 일

1. `merged_dataset.csv`를 `data/` 에 배치
2. 데이터 계약서 대비 검증 (컬럼, dtype, 유일키, 결측 형식)
3. `notebooks/eda_template.ipynb` 전체 실행
4. baseline 성능 측정 → 팀 공유
5. 주요 비율 컬럼 Top 후보 정리
