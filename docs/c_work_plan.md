# C 담당자 작업 계획서 — Tree-based 모델링

## 전제 조건

- 전처리된 데이터셋: `preprocess/data/processed/H{6,8,10,12,14,16,18,20,22,24}/` 에 train/valid/test.csv 존재
- 피처 41개 컬럼: meta 4개(stock_code, year, quarter, gics_sector) + 재무비율 24개 + 원시값 5개(유형자산, 무형자산, 상각비 3종) + 매크로 6개 + label
- 전처리 완료 사항: ffill → CF=0 → 섹터·분기 중앙값 보간 → 이상치 클리핑
- 극심한 클래스 불균형: H10 기준 train_pos=193 / train_rows=30,764 (imbalance_ratio ≈ 158:1)
- baseline 존재: `src/baseline/run_baseline.py` (LogisticRegression, DecisionTree — class_weight="balanced")

---

## 완료 상태 확인

| 항목 | 상태 | 비고 |
|---|---|---|
| 학습용 데이터셋 입력 포맷 점검 (train/valid/test) | **완료** | H6~H24 총 10개 horizon, 연도 기반 time split |
| 수치형/범주형 피처 분리 및 전처리 기준 확정 | **완료** | `classify_columns()` — ratio/raw_value/macro 분류 |
| 결측치 처리 방식 정의 | **완료** | 섹터·분기 중앙값 보간 + 클리핑, meta.json에 기준값 저장 |
| 클래스 불균형 대응 전략 적용 | **완료** | 7전략 비교 → baseline+threshold 최적화 확정, sampling 불채택 |
| Random Forest baseline 학습 | **완료** | H10/H12 학습 완료, PR-AUC 1위 |
| Gradient Boosting 계열 baseline 학습 | **완료** | H10/H12 학습 완료 |
| XGBoost 실험 | **완료** | H10/H12 학습 완료, early stopping 적용 |
| LightGBM 실험 | **완료** | H10/H12 학습 완료, early stopping 적용 |
| CatBoost 실험 여부 검토 | **비채택** | gics_sector가 meta 컬럼이라 네이티브 범주형 이점 없음 |
| 모델별 주요 metric 비교 (F1, ROC-AUC, PR-AUC, Recall) | **완료** | H10/H12 × 4모델, `results/comparison_test.csv` |
| Feature Importance 추출 | **완료** | 4모델 평균, `results/feature_importance_H{10,12}.csv` |
| 결과 리포트 정리 및 다음 실험 방향 설정 | **완료** | `results/summary.md` |

---

## 작업 순서 전체 흐름

```
Phase 1: 실험 인프라 구축 → Phase 2: 불균형 전략 확정 → Phase 3: 모델 학습 → Phase 4: 분석 & 리포트
```

Phase 1이 완료되어야 Phase 2~4에서 일관된 실험 프레임워크 위에서 진행 가능.

---

## Phase 1: 실험 인프라 구축

### 목표
H-horizon별 데이터를 로드하고, 모델 학습/평가를 일관되게 수행할 수 있는 실험 파이프라인 구축

### 작업 단계

#### 1-1. 데이터 로더 구현
- `preprocess/data/processed/H{n}/train.csv`, `valid.csv`, `test.csv` 로드 함수
- meta 컬럼(stock_code, year, quarter, gics_sector) 제외 후 X/y 분리
- gics_sector 인코딩 방식 결정 (Label Encoding vs One-Hot vs 제외)

#### 1-2. 평가 프레임워크 통일
- 기존 `src/baseline/run_baseline.py`의 `evaluate()` 확장
- 주요 metric: **F1, Precision, Recall, ROC-AUC, PR-AUC**
- PR-AUC를 primary metric으로 설정 (극심한 불균형 시 ROC-AUC는 낙관적)
- threshold 튜닝 로직 추가 (PR curve 기반 최적 threshold 탐색)

#### 1-3. 실험 결과 저장 구조
- 산출물 경로: `results/` 디렉터리
- 모델별 결과 JSON: `results/{model_name}_{horizon}.json`
- 비교 테이블 자동 생성 함수

### 예상 소요
- 1-1: 30분
- 1-2: 1시간
- 1-3: 30분

---

## Phase 2: 클래스 불균형 대응 전략 확정

### 목표
imbalance_ratio ≈ 158:1 환경에서 최적의 불균형 대응 조합 결정

### 작업 단계

#### 2-1. 전략 후보 목록

| 전략 | 구현 방식 | 비고 |
|---|---|---|
| class_weight | `class_weight="balanced"` (sklearn 내장) | baseline에 이미 적용 |
| scale_pos_weight | XGBoost/LightGBM 파라미터 | neg/pos 비율 직접 지정 |
| SMOTE | `imblearn.over_sampling.SMOTE` | 소수 클래스 합성 오버샘플링 |
| SMOTE + Tomek | `imblearn.combine.SMOTETomek` | 오버샘플링 + 경계 정리 |
| Random UnderSampling | `imblearn.under_sampling` | 다수 클래스 축소 |
| Threshold 조정 | PR curve 기반 최적 threshold | 후처리 방식 |

#### 2-2. 불균형 전략 비교 실험
- RandomForest를 기준 모델로 선정
- 위 전략 조합별 valid set PR-AUC 비교
- 최적 전략 1~2개 확정 → Phase 3에 적용

#### 2-3. 결측치 잔존 여부 확인
- 전처리된 데이터에 NaN 잔존 여부 최종 점검
- 잔존 시 처리 방식: median imputation (sklearn SimpleImputer)

### 예상 소요
- 2-1: 20분
- 2-2: 1.5시간
- 2-3: 20분

---

## Phase 3: Tree-based 모델 학습

### 목표
5개 모델(RF, GBM, XGBoost, LightGBM, CatBoost)을 H-horizon별로 학습하고 비교

### 디렉터리 구조

```
src/
├── baseline/
│   └── run_baseline.py          # 기존 LR + DT
└── modeling/
    ├── __init__.py
    ├── data_loader.py           # Phase 1-1
    ├── evaluate.py              # Phase 1-2
    ├── train_rf.py              # Phase 3-1
    ├── train_gbm.py             # Phase 3-2
    ├── train_xgboost.py         # Phase 3-3
    ├── train_lightgbm.py        # Phase 3-4
    ├── train_catboost.py        # Phase 3-5 (선택)
    └── run_all.py               # 전체 실행 스크립트

results/
├── comparison_H10.csv
├── rf_H10.json
├── ...
└── summary.md
```

### 작업 단계

#### 3-1. Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
```

- 하이퍼파라미터 초기값 설정 후 학습
- valid set으로 평가
- feature_importances_ 추출

#### 3-2. Gradient Boosting (sklearn)

```python
from sklearn.ensemble import GradientBoostingClassifier

GradientBoostingClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42,
)
```

- sklearn GBM은 class_weight 미지원 → sample_weight로 대체
- 학습 속도가 느리므로 XGBoost/LightGBM 대비 우선순위 낮음

#### 3-3. XGBoost

```python
import xgboost as xgb

xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=ratio,  # neg/pos
    eval_metric="aucpr",
    early_stopping_rounds=50,
    random_state=42,
    n_jobs=-1,
)
```

- `scale_pos_weight` = (num_negative / num_positive) 자동 계산
- early stopping으로 과적합 방지
- eval_metric을 `aucpr`로 설정 (불균형 대응)

#### 3-4. LightGBM

```python
import lightgbm as lgb

lgb.LGBMClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    is_unbalance=True,
    metric="average_precision",
    early_stopping_rounds=50,
    random_state=42,
    n_jobs=-1,
)
```

- `is_unbalance=True` 또는 `scale_pos_weight` 사용
- 범주형 피처(gics_sector) 네이티브 지원 가능
- 학습 속도 가장 빠름 → 하이퍼파라미터 탐색 기준 모델로 적합

#### 3-5. CatBoost 실험 여부 검토

**채택 기준:**
- LightGBM/XGBoost 대비 유의미한 PR-AUC 개선 가능성
- 범주형 피처(gics_sector, quarter) 네이티브 인코딩 이점
- 학습 시간 대비 효용

**검토 결과에 따라:**
- 채택 시: `train_catboost.py` 구현
- 비채택 시: 사유 기록 후 Phase 4에서 LightGBM/XGBoost에 집중

### H-horizon 실험 범위
- 우선: H10, H12 (가장 실용적인 예측 기간)
- 확장: H6, H8, H14~H24 (패턴 비교용)
- 각 horizon별 train/valid/test는 meta.json에 정의된 연도 기준 사용

### 예상 소요
- 3-1: 1시간
- 3-2: 30분
- 3-3: 1.5시간
- 3-4: 1.5시간
- 3-5: 1시간 (검토 포함)

---

## Phase 4: 분석 & 리포트

### 목표
모델별 성능 비교, Feature Importance 추출, 다음 실험 방향 설정

### 작업 단계

#### 4-1. 모델별 주요 metric 비교

| Model | F1 | Precision | Recall | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| LogisticRegression | - | - | - | - | - |
| DecisionTree | - | - | - | - | - |
| RandomForest | - | - | - | - | - |
| GradientBoosting | - | - | - | - | - |
| XGBoost | - | - | - | - | - |
| LightGBM | - | - | - | - | - |
| (CatBoost) | - | - | - | - | - |

- valid set 기준으로 모델 선정 → test set으로 최종 보고
- H-horizon별 성능 변화 트렌드 분석

#### 4-2. Feature Importance 추출

- tree-based 모델: `feature_importances_` (Gini / Gain)
- XGBoost/LightGBM: gain, weight, cover 세 가지 importance
- 상위 15개 피처 시각화 (horizontal bar plot)
- 모델 간 importance 순위 비교 → 공통 핵심 피처 도출

#### 4-3. 전처리 옵션별 비교 (선택)

`build_h_datasets.py`에 정의된 4가지 실험 설정:

| 설정 | clipping | Winsorize | RobustScaler |
|---|---|---|---|
| baseline | O | X | X |
| exp-A | O | O | X |
| exp-B | O | X | O |
| exp-C | O | O | O |

- 최적 모델 1~2개에 대해 전처리 옵션 비교

#### 4-4. 결과 리포트 정리

- **산출물**: `results/summary.md`
- 포함 내용:
  - 최적 모델 및 하이퍼파라미터
  - H-horizon별 최적 성능
  - 핵심 피처 Top 10
  - 불균형 전략 효과 비교
  - 남은 과제 및 다음 실험 방향

#### 4-5. 다음 실험 방향 설정

후보:
- 하이퍼파라미터 튜닝 (Optuna / GridSearch)
- 피처 엔지니어링 (교차 비율, 시계열 lag/diff)
- 앙상블 (Stacking, Blending)
- 임계값(threshold) 최적화
- SHAP 기반 모델 해석

### 예상 소요
- 4-1: 1시간
- 4-2: 1시간
- 4-3: 1시간 (선택)
- 4-4: 1시간
- 4-5: 30분

---

## 일정 요약

| 순서 | 작업 | 산출물 | 예상 소요 |
|---|---|---|---|
| 1 | 실험 인프라 구축 | `src/modeling/data_loader.py`, `evaluate.py` | 2시간 |
| 2 | 불균형 전략 확정 | 전략 비교 결과 (valid PR-AUC 기준) | 2시간 |
| 3 | RF 학습 | `train_rf.py`, `results/rf_*.json` | 1시간 |
| 4 | GBM 학습 | `train_gbm.py`, `results/gbm_*.json` | 30분 |
| 5 | XGBoost 학습 | `train_xgboost.py`, `results/xgb_*.json` | 1.5시간 |
| 6 | LightGBM 학습 | `train_lightgbm.py`, `results/lgbm_*.json` | 1.5시간 |
| 7 | CatBoost 검토 | 채택/비채택 사유 | 1시간 |
| 8 | metric 비교 & Importance | 비교 테이블, importance plot | 2시간 |
| 9 | 리포트 & 다음 방향 | `results/summary.md` | 1.5시간 |

**합계: 약 13시간 (넉넉히 2일)**

---

## 생성할 파일 목록

```
src/modeling/
├── __init__.py
├── data_loader.py               # Phase 1-1
├── evaluate.py                  # Phase 1-2
├── train_rf.py                  # Phase 3-1
├── train_gbm.py                 # Phase 3-2
├── train_xgboost.py             # Phase 3-3
├── train_lightgbm.py            # Phase 3-4
├── train_catboost.py            # Phase 3-5 (선택)
└── run_all.py                   # 전체 실행

results/
├── comparison_H10.csv           # Phase 4-1
├── feature_importance_H10.png   # Phase 4-2
└── summary.md                   # Phase 4-4
```

---

## 추가 필요 패키지 (requirements.txt)

```
xgboost>=2.0.0
lightgbm>=4.0.0
catboost>=1.2.0        # Phase 3-5 채택 시
imbalanced-learn>=0.11.0  # SMOTE 등 sampling 전략 사용 시
optuna>=3.3.0          # 후속 하이퍼파라미터 튜닝 시
shap>=0.43.0           # 후속 모델 해석 시
```

---

## 핵심 주의사항

1. **PR-AUC를 primary metric으로** — 158:1 불균형에서 ROC-AUC는 과대평가 위험
2. **valid set으로 모델 선정, test set은 최종 보고용** — data leakage 방지
3. **H-horizon별 meta.json의 전처리 기준값 사용** — train에서 fit한 기준으로 valid/test transform (이미 전처리 완료 상태)
4. **gics_sector 인코딩 주의** — LightGBM/CatBoost는 네이티브 범주형 지원, sklearn 계열은 인코딩 필요
5. **전처리 옵션(Winsorize/RobustScaler) 실험은 Phase 4-3에서** — 현재 데이터는 clipping만 적용된 baseline 버전
