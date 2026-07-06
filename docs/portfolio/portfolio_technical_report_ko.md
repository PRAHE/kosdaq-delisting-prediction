# 한국 상장폐지 조기경보 모델 — 진단·시계열·외부 데이터를 통한 통계적 천장 돌파

## Technical Report (Portfolio 통합)

**저자**: 이현지 (광운대학교 컴퓨터정보공학부)
**연락**: wendydeer@naver.com
**작성일**: 2026-06-01
**지원 연구실**: UNIST Data Analytics Lab (김성일 교수)

본 보고서는 한국 상장폐지 조기경보 예측 프로젝트의 진단(S0, SR), 시계열 모델링(S1 Phase 1~3 + walk-forward), 외부 데이터 통합(A1 감사의견, A2 관리종목, A3 시장), 그리고 천장 돌파의 통계적 입증(Best CI)을 하나의 narrative로 통합한 학부 portfolio 문서다. 각 단계의 상세 결과는 `docs/research_*_report.md`의 8개 별도 보고서를 참고할 수 있다.

---

## 목차

1. [초록 (Abstract)](#1-초록-abstract)
2. [서론 — 문제 정의와 출발점](#2-서론--문제-정의와-출발점)
3. [데이터와 Baseline 재현](#3-데이터와-baseline-재현)
4. [진단 1 — baseline의 신뢰구간 (S0)](#4-진단-1--baseline의-신뢰구간-s0)
5. [진단 2 — 라벨링 정의 점검 (SR)](#5-진단-2--라벨링-정의-점검-sr)
6. [시계열 모델링 (S1: Phase 1~3 + Walk-Forward)](#6-시계열-모델링-s1-phase-13--walk-forward)
7. [외부 데이터 통합 (A3, A1, A2)](#7-외부-데이터-통합-a3-a1-a2)
8. [천장 돌파의 통계적 입증 (Best CI)](#8-천장-돌파의-통계적-입증-best-ci)
9. [종합 및 해석](#9-종합-및-해석)
10. [한계와 후속 연구](#10-한계와-후속-연구)
11. [재현성 (Reproducibility)](#11-재현성-reproducibility)
12. [참고 문헌](#12-참고-문헌)
13. [부록 — 코드 구조 및 산출물 목록](#13-부록--코드-구조-및-산출물-목록)

---

## 1. 초록 (Abstract)

### English

Prior Korean stock-delisting early-warning models on quarterly financial-ratio
snapshots had stalled at **PR-AUC ≈ 0.29** through nine reported improvement
attempts (RF/XGB/LGBM tuning, SMOTE, feature engineering, alternative labels),
with the source of stagnation undiagnosed. This study applies a diagnostic-
first methodology — Bayesian bootstrap (Rubin 1981) confidence intervals,
walk-forward cross-validation with multi-seed measurement, paired statistical
testing, multivariate-MMD distribution-shift detection — and finds that the
0.2876 figure sits inside a wide 95% Bayesian credible interval **[0.180,
0.421]**, with **all nine prior experiments statistically indistinguishable**
from baseline. Building on this base, an irregular-time-series model
(Transformer + Time2Vec, K=3) combined with external audit-opinion features
(DART OpenAPI, 16,896 calls; 5 features encoding KOSDAQ Listing Regulation §38
triggers) lifts the model to **PR-AUC 0.432 ± 0.026** on the original single
split (5 seeds). A paired Bayesian bootstrap (n=5,000) on shared Dirichlet
weights shows **Δ 95% CI [+0.094, +0.279]** — strictly positive — and
**P(best > baseline) = 1.000**, statistically establishing the breakthrough.
The empirical finding replicates Geiger & Raghunandan (2002) on Korean data:
two consecutive non-clean audit opinions yield a 96× positive-rate lift.

### 한국어

본 연구는 한국 코스닥/코스피 상장폐지 조기경보 모델의 보고된 PR-AUC 천장값
**0.2876**을 통계적 엄밀성으로 재평가하고, 그것이 실제로 깨질 수 있는지를
검증한다. 진단 단계에서 **Bayesian bootstrap**으로 baseline PR-AUC의 95%
신뢰구간이 **[0.180, 0.421]**이며 폭이 0.24에 달하는 점, exp_010~018의 9개
기존 실험이 모두 이 구간 안의 noise로 통계적으로 구분되지 않는 점을 확인했다.
이를 기반으로 **Transformer + Time2Vec**과 **DART 감사의견** 외부 데이터
(5 features, KOSDAQ 상장규정 38조 트리거 인코딩)를 결합하여 PR-AUC를
**0.432 ± 0.026**으로 끌어올렸다. 같은 Dirichlet 가중치를 공유하는
paired Bayesian bootstrap(n=5,000)에서 **Δ의 95% CI [+0.094, +0.279]**가
모두 양수 영역에 있고 **P(best > baseline) = 1.000**으로 천장 돌파가
통계적으로 확립되었다. Geiger & Raghunandan(2002)의 감사의견 신호가 한국
코스닥에서 강하게 재현되며(비적정 연속 2회 → 양성 비율 96배), 시장 변수
(Campbell, Hilscher & Szilagyi 2008)는 부분적으로만 재현됨도 확인했다.

> **하이라이트 그림**: `docs/portfolio/hero_figure.png` — 본 보고서 전체 narrative를 한 장에 요약.

---

## 2. 서론 — 문제 정의와 출발점

### 2.1 한국 상장폐지 조기경보 문제

한국 주식시장에서 상장폐지는 투자자에게 회복 불가능한 손실을 입히는 가장
치명적인 사건이다. 그러나 정보력과 분석 능력이 뛰어난 기관·외국인 투자자와
달리 대다수 개인투자자는 정보 접근성이 떨어져 리스크를 그대로 떠안는다.
재무제표·공시·거시경제 지표를 결합하여 **상폐 1년 전 시점에서 위험을 식별**
하는 모델은 개인투자자의 보호와 시장 공정성 확보에 직접 기여할 수 있다.

본 프로젝트는 DART(전자공시시스템) 재무제표 + 거시경제 지표를 활용한 분기
단위 예측 모델을 빌드한다. 데이터·전처리·이전 9개 실험은 이미 팀에서
구축되어 있었고, 본 연구는 그 위에서 시작한다.

### 2.2 출발점 — "0.2876 천장"의 정체

선행 9개 실험(`exp_010`~`exp_018`)은 다음과 같다:

| 실험 | 개입 | Test PR-AUC |
|------|------|-------------|
| exp_010 | Threshold 전략 (F1, F2, recall-fixed) | 0.2876 |
| exp_011 | RF Optuna 50 trials | 0.2876 |
| exp_012 | XGB Optuna 50 trials | 0.2078 |
| exp_013 | Feature engineering (missing/YoY/distress) | 0.2876 |
| exp_014 | RF class_weight | 0.2876 |
| exp_015 | LGBM Optuna 50 trials | 0.1930 |
| exp_016 | SMOTE 오버샘플링 | 0.2876 |
| exp_017 | Piotroski F-score | 0.2876 |
| exp_018 | fixed_N2 라벨 (2년 전) | 0.2876 |

모두 baseline RF PR-AUC **0.2876**을 못 넘긴 상태에서, *"천장은 0.29인가
혹은 평가 노이즈가 큰 것인가"* 가 미해결이었다. 본 연구의 첫 질문은 바로
이것이다.

### 2.3 본 연구의 5가지 contribution

1. **진단 방법론 (S0)**: Bayesian bootstrap CI + paired bootstrap 통계 비교
   + Holm-Bonferroni 보정 + 다변량 MMD 분포 이동 검정 + walk-forward CV를
   본 문제에 처음 통합 적용.
2. **Negative-result rigor**: 9개 사전 실험을 통계적으로 baseline의 95% CI
   안에 있는 "noise"로 재해석.
3. **시계열 모델링 진화**: GRU → GRU-D → Transformer + Time2Vec → Neural
   CDE 4단계 ablation. Transformer가 walk-forward 평균 PR-AUC 0.289로
   single-snapshot 한계를 넘어섰다.
4. **이론의 한국 재현**: Geiger & Raghunandan(2002)의 감사의견 부도 예측력이
   한국 코스닥에서 96배 lift로 강하게 재현. Campbell+(2008) 시장 변수는
   부분적으로만 재현.
5. **통계적 천장 돌파**: paired bootstrap Δ CI [+0.094, +0.279], P(best >
   baseline) = 1.000. 5,000회 resampling 중 baseline이 best를 이긴 경우 0.

---

## 3. 데이터와 Baseline 재현

### 3.1 패널 데이터

| 항목 | 값 |
|------|-----|
| 종목 수 | 1,536 unique (KOSPI + KOSDAQ) |
| 연도 범위 | 2015 ~ 2025 (train 2015~2022, valid 2023, test 2024) |
| 분기 단위 | Q1, H1, Q3, ANNUAL 혼재 |
| 총 행수 | 45,854 (stock × year × quarter) |
| 양성 (fixed_N1) | train 283, valid 39, test 56 (test base rate ~1%) |
| 클래스 불균형 | 124 : 1 |
| 피처 (기본) | 33개 = 27 재무비율 + 3 YoY 증가율(결측 많아 제외) + 6 거시경제 |

피처 예시: 부채비율, 자기자본비율, 매출액순이익률, 매출총이익률, 현금비율,
유동비율, 당좌비율, 차입금의존도, 재고자산회전율, 매출원가율, vix_avg,
gdp_growth_yoy, credit_spread 등.

### 3.2 라벨링 — fixed_N1

`fixed_N1`: 상폐일 기준 **정확히 1년 전 연도**의 모든 분기 행을 label=1.

- 양성 정의: 회계연도 Y에 발표된 데이터 → 다음 해 Y+1에 상폐가 일어났으면 label=1
- 합병·자진상폐 같은 구조적 상폐는 제외 (재무·회계 기반 위험만)
- 양성 비율 train 0.8%, test 1.05% — 극심한 불균형

### 3.3 Canonical RF Baseline

선행 실험에서 보고된 baseline:

```python
RandomForestClassifier(
    n_estimators=200, max_depth=10,
    min_samples_leaf=5, max_features="sqrt",
    random_state=42, n_jobs=-1,
)
# 전처리: SimpleImputer(median) → signed_log1p
```

본 연구 첫 단계에서 이 설정을 재현하여 **PR-AUC = 0.2876, ROC-AUC =
0.8586** 정확 일치 확인. 이후 모든 비교의 anchor가 된다.

---

## 4. 진단 1 — baseline의 신뢰구간 (S0)

전체 보고서: `docs/research_s0_diagnostic_report.md`.

### 4.1 6개 진단 실험 구성

| ID | 진단 | 답하려는 질문 |
|----|------|---------------|
| A1 | Bayesian + Frequentist bootstrap CI | 0.2876의 통계적 신뢰구간은? |
| A2 | Walk-forward CV (4 fold × 5 seed) | 평가 안정성이 얼마나 흔들리나? |
| A3 | Paired bootstrap vs exp_010~018 | 9개 실험은 baseline과 통계적으로 다른가? |
| A4 | PSI/KL/JS + 다변량 MMD permutation | 분포 이동(2023→2024)은 어디에? |
| A5 | 비지도 이상탐지 5종 (IsoForest/LOF/OCSVM/HBOS/ECOD) | 본 문제의 본질은 anomaly detection인가? |
| A6 | SHAP + Permutation Importance + Calibration | baseline은 어디서 틀리는가? |

### 4.2 핵심 발견 5가지

**(1) baseline의 95% CI는 매우 넓다.**

test 양성 56개에 대한 Bayesian bootstrap (n=2,000):

| Metric | 점추정 | 평균 | 95% CI | std |
|--------|--------|------|---------|------|
| PR-AUC | 0.2876 | 0.2938 | **[0.180, 0.421]** | 0.062 |
| ROC-AUC | 0.8586 | 0.8589 | [0.807, 0.902] | 0.025 |

PR-AUC의 95% 신뢰구간 **폭 0.24**. ROC-AUC는 CI 폭 0.10으로 ranking은 안정.
즉 "0.2876"이라는 점추정은 매우 넓은 분포의 한 점.

**(2) 9개 사전 실험이 모두 baseline의 noise band 안.**

baseline posterior 위에서 paired Bayesian 비교, Holm-Bonferroni 보정:

| 실험 | best variant | PR-AUC | Δ | 95% CI 안 | Holm 보정 p |
|------|--------------|--------|---|-----------|-------------|
| exp_010 | F1 strategy | 0.2876 | 0 | 예 | 1.000 |
| exp_011 | RF Optuna | 0.2876 | 0 | 예 | 1.000 |
| exp_012 | XGB tuned_f1 | 0.2078 | −0.080 | 예 | 1.000 |
| exp_013 | FE Step0 | 0.2876 | 0 | 예 | 1.000 |
| exp_014 | class_weight None | 0.2876 | 0 | 예 | 1.000 |
| exp_015 | LGBM tuned_f1 | 0.1930 | −0.095 | 예 | 1.000 |
| exp_016 | SMOTE None | 0.2876 | 0 | 예 | 1.000 |
| exp_017 | Piotroski | 0.2876 | 0 | 예 | 1.000 |
| exp_018 | fixed_N1 | 0.2876 | 0 | 예 | 1.000 |

**Holm 보정 후 p < 0.05인 실험 0건**. "성능이 떨어진" 실험들은 더 못한
모델이 아니라 같은 분포에서 추출된 무작위 표본이다.

**(3) 2023 valid는 우연히 가장 어려운 연도.**

| Fold | Valid 연도 | RF Valid PR-AUC mean |
|------|-----------|----------------------|
| 1 | 2020 | 0.192 |
| 2 | 2021 | 0.239 |
| 3 | 2022 | 0.239 |
| 4 | **2023** | **0.116** ← 우리가 anchor로 쓰던 그 연도 |

→ 가장 어려운 valid 연도를 hyperparameter 선택 anchor로 써왔던 것.

**(4) 분포 이동의 정체는 거시변수.**

피처별 PSI(train→test):
- 거시변수 6개(VIX, CPI, USDKRW 등) PSI > 7 (연도 키라서 구조적)
- 재무비율은 대부분 PSI < 0.10 (안정적)
- 부채비율만 PSI 1.45 (유일 financial outlier)

다변량 MMD(RBF, permutation):
- train vs test: MMD² = 0.016 (p = 0)
- valid vs test: MMD² = 0.017 (p = 0) ← train→valid보다 큼

S0의 §9에서 본 연구는 "거시 shift ≠ 신호 shift"로 결론. permutation
importance에서 거시변수가 신호 거의 0 (vix_avg ~0.002, 현금비율의 1/20).

**(5) supervised label에는 비지도가 복원 못 하는 정보가 있다.**

| 모델 | Test PR-AUC | Δ vs supervised RF |
|------|-------------|---------------------|
| RF (supervised) | **0.288** | 0 |
| HBOS | 0.160 | −0.127 |
| IsolationForest | 0.109 | −0.179 |
| OCSVM | 0.086 | −0.201 |
| LOF | 0.036 | −0.251 |

→ "순수 anomaly detection으로 가도 안 된다" 결론. 라벨이 가진 정보가
중요하며, 비지도 시그널은 보조적.

### 4.3 calibration & 모델 해석 (A6)

- ECE = **0.0026**, Brier = 0.0092 — 모델은 매우 잘 보정되어 있음
- threshold 0.097에서: TP=21, FP=42, FN=35
- Top permutation importance: **현금비율, 순운전자본비율, 매출원가율,
  매출채권회전율, 당좌비율** — 유동성/수익성 중심
- 거시변수는 거의 신호 없음

**남은 오차의 본질은 35개의 confident FN** — 상폐 1년 전인데도 재무비율상
정상으로 보이는 기업들. 시계열 + 외부 정보가 필요한 근거.

---

## 5. 진단 2 — 라벨링 정의 점검 (SR)

전체 보고서: `docs/research_sr_labeling_report.md`.

### 5.1 의문 — "라벨 1이 너무 적어서가 아닌가?"

회의에서 "라벨 1 데이터의 절대 부족(124:1)이 근본 문제이고, '상폐 예측'을
'상폐 위험 검출'로 비틀면 라벨 1을 확장할 수 있지 않을까"라는 의견이 제기됨.

### 5.2 7가지 라벨링 비교 (train 라벨링만 변경, valid/test = L1 fixed_N1 고정)

| ID | 라벨링 | Train 양성 | Test PR-AUC (5-seed mean) |
|----|--------|-----------|----------------------------|
| L1 | fixed_N1 (현재 baseline) | 262 | 0.257 |
| L2 | rolling_H12 (1년 내) | 348 | 0.234 |
| **L3** | **rolling_H24 (2년 내)** | **641** | **0.263** ← 1위 |
| L4 | fixed_N1 ∪ N2 ∪ N3 (1·2·3년 전 모두) | 871 | 0.256 |
| L5 | ordinal 4-class (정상/N3/N2/N1) | full | 0.252 |
| L6 | continuous risk `exp(-Δt/2)` | full | 0.259 |
| L7 | v3-style 상폐 기업 모든 연도 = 1 | 2,094 | 0.245 |

### 5.3 핵심 발견 3가지

1. **7개 모두 baseline 95% CI [0.180, 0.421] 안** — 라벨링 변경만으로는
   천장 못 깸.
2. **L3 rolling_H24가 가장 일관**: PR-AUC, top-20 precision, walk-forward
   안정성 모두 1위. 이후 모든 시계열 학습의 default train 라벨.
3. **v3 사례 재해석**: "valid 0.42 → test 0.234 붕괴"는 라벨링 자체의 문제가
   아니라 **valid를 v3 라벨로 평가하면서 양성 base rate가 부풀려진 착시**.
   train=v3, valid/test=L1로 통일하면 test PR-AUC가 baseline 수준에 머물고
   붕괴가 사라짐. v3는 "잘못된 라벨링"이 아니라 "평가 일관성 부재".

### 5.4 결론

- 라벨링은 **소폭의 일관된 개선 여지를 제공하지만 천장 자체는 못 깸**.
- 다음 stream은 라벨링 추가 변형이 아니라 **시계열 모델링(S1) 또는 hazard
  모델(S3)** 로 가야 한다.
- 단 모든 후속 stream에서 train 라벨은 **L3 rolling_H24**를 default로 채택.

---

## 6. 시계열 모델링 (S1: Phase 1~3 + Walk-Forward)

전체 보고서: `docs/research_s1_phase1_report.md` ~ `phase3_report.md`,
`walkforward_report.md`.

### 6.1 동기 — S0의 35개 confident FN

S0-A6에서 본 35개 FN은 "상폐 1년 전인데도 재무비율상 정상으로 보이는 기업"
이다. 단일 스냅샷으로는 잡을 수 없으며, **과거 3~5년 trajectory(악화 패턴)**
이 필요하다. Beaver(1966)가 부도 기업의 5년 추세를 핵심 신호로 보였던 것의
한국 재현이다.

### 6.2 4-Phase 설계

| Phase | 모델 | 근거 |
|-------|------|------|
| 1 | GRU | Cho et al. (2014) EMNLP — 가장 단순한 sequence baseline |
| 2 | GRU-D | Che et al. (2018) *Scientific Reports* — feature-level missing decay |
| 3 | Transformer + Time2Vec | Vaswani et al. (2017) + Kazemi et al. (2019) — self-attention + 학습 가능 시간 임베딩 |
| 4 (계획) | Neural CDE | Kidger et al. (2020) NeurIPS — 불규칙 시점 처리, UNIST DAL 시그니처 |

데이터 구성: 각 (stock_code, year, quarter) target에 대해 **동일 quarter
타입의 과거 3년 시퀀스** (K=3). 결측 mask 함께. focal loss(γ=2, α=0.25).

### 6.3 Phase 1 — GRU baseline

1-layer GRU (hidden=64) + mean pooling + 선형 head. focal loss + Adam +
ReduceLROnPlateau, 5 seed.

**결과 (단일 split test 2024 5-seed mean):**

| 모델 | Test PR-AUC mean ± std | ROC-AUC | P@20 |
|------|--------------------------|---------|------|
| GRU (Phase 1) | **0.304 ± 0.046** | 0.853 | 0.560 |
| RF baseline | 0.270 ± 0.015 | 0.855 | 0.580 |

GRU가 +0.034 우위지만 std가 RF의 3배. 5-seed ensemble bootstrap point
0.376 [0.257, 0.513]로 보면 +0.106 우위. **시계열 입력의 효과는 실재**.

### 6.4 Phase 2 — GRU-D (Negative Result)

가장 결측이 많은 데이터를 다루는 방법으로 GRU-D 시도. `combined_raw.csv`에서
NaN 보존, feature-level mask + time gap δ + input/hidden decay 학습.

**결과:**

| 모델 | Test PR-AUC mean ± std | ROC-AUC |
|------|--------------------------|---------|
| GRU-D (Phase 2) | 0.267 ± 0.068 | **0.875** |
| GRU (Phase 1, 비교용) | 0.304 ± 0.046 | 0.853 |

**Negative result**: GRU-D가 GRU보다 mean −0.037, std 1.5배. 추가
파라미터 ~2,242개가 양성 641개에서 overfit. 다만 ROC-AUC는 1위.
"feature-level decay" 자체는 본 데이터 규모에서 가성비 나쁨.

### 6.5 Phase 3 — Transformer + Time2Vec (New Best)

Time2Vec(τ_0(t) = ω_0·t + φ_0, τ_i(t) = sin(ω_i·t + φ_i)) → input projection
→ 1-layer Transformer encoder (model_dim=64, heads=4, GELU, pre-norm) →
mask-aware mean pooling → MLP head.

**결과 (단일 split test, 5-seed):**

| 모델 | Test PR-AUC mean ± std | ROC-AUC | P@20 | P@50 |
|------|--------------------------|---------|------|------|
| **Transformer (Phase 3)** | **0.342 ± 0.023** | **0.889** | **0.70** | **0.388** |
| GRU (Phase 1) | 0.304 ± 0.046 | 0.853 | 0.56 | 0.380 |
| GRU-D (Phase 2) | 0.267 ± 0.068 | 0.875 | 0.56 | 0.328 |
| RF baseline | 0.269 ± 0.015 | 0.855 | 0.58 | 0.368 |

**Transformer가 5-seed mean / ROC-AUC / P@20 / P@50 모든 지표 1위**. std는
0.023으로 GRU의 절반 — **학습 안정성도 함께 개선**. P@20=0.70은 top 20 중
14개 양성 적중 (다른 모델 11~12개).

**핵심 추측**:
- Time2Vec가 명시적 시간 임베딩 제공 → "1년 차이 ≠ 2년 차이"를 학습
- Pre-norm + GELU 안정성
- Mean pooling이 마지막 step over-reliance 회피

### 6.6 Walk-Forward 재검증 (4 fold × 3 seed)

Phase 3 우위가 단일 split (valid 2023이 가장 어려운 연도)의 우연인지
검증. S0-A2와 같은 walk-forward 구성.

**Fold별 test PR-AUC (3-seed mean):**

| Fold | Valid | GRU | Transformer | Δ |
|------|-------|-----|-------------|---|
| 1 | 2020 | 0.092 | **0.222** | **+0.130** |
| 2 | 2021 | 0.190 | **0.277** | +0.087 |
| 3 | 2022 | 0.259 | **0.313** | +0.054 |
| 4 | 2023 | 0.309 | **0.343** | +0.034 |
| **평균** | — | **0.213** | **0.289** | **+0.076** |

**Transformer 4/4 fold 모두 우위.** Fold 1(가장 적은 train 데이터)에서 +0.130,
fold 4(가장 많은 train)에서 +0.034 — **train 데이터 적을수록 Transformer
우위가 큼**. 이는 작은 데이터에서 Time2Vec + attention의 정규화 효과가
GRU의 hidden state 의존 학습보다 robust함을 시사.

**전체 평균:**

| 모델 | Test PR-AUC | std | ROC-AUC | P@20 | P@50 |
|------|-------------|-----|---------|------|------|
| **Transformer+T2V** | **0.289** | **0.056** | **0.887** | **0.575** | **0.338** |
| GRU | 0.213 | 0.094 | 0.864 | 0.371 | 0.285 |
| Δ | +0.076 | −0.038 | +0.023 | +0.204 | +0.053 |

Transformer가 std는 GRU의 60% 수준, P@20은 +0.20 압승. **Phase 3 우위는
단일 split의 우연이 아닌 robust한 결과**임이 통계적으로 확립.

### 6.7 GRU의 fold 1 overfit 패턴

GRU의 fold 1 valid PR-AUC = 0.399, test PR-AUC = 0.092. **격차 +0.307**.
같은 fold에서 Transformer는 valid 0.275, test 0.222로 격차 +0.053 (50배
안정). **시간 임베딩이 분포 이동에 더 robust**함을 보여줌.

---

## 7. 외부 데이터 통합 (A3, A1, A2)

### 7.1 A3 — 주가·거래량 시장 피처

전체 보고서: `docs/research_a3_market_features_report.md`.

**데이터**: FinanceDataReader로 1,945종목 중 1,516종목의 일별 OHLCV
(2014~2025) 수집. 분기 말 시점 기준 6개 피처 산출.

| 피처 | 정의 | 관측률 |
|------|------|--------|
| price_log_close | 분기 말 종가 로그 | 100% |
| price_ret_12m | 12개월 수익률 | 94.9% |
| price_volatility_60d | 60일 변동성 (annualized) | 99.8% |
| price_drawdown_max_12m | 12개월 최고가 대비 낙폭 | 98.9% |
| volume_log_mean_60d | 60일 평균 거래량 로그 | 99.9% |
| volume_change_yoy | 거래량 YoY 변화율 | 92.7% |

이론 근거: Campbell, Hilscher & Szilagyi(2008) — 시장 변수가 회계 변수보다
부실 예측력 높음. Bharath & Shumway(2008) — Merton DD model 핵심 입력.

**결과 (Transformer walk-forward, 평균 12 측정):**

| 지표 | 33 feat | 39 (+ market) | Δ |
|------|---------|----------------|---|
| Test PR-AUC | 0.289 | 0.297 | +0.008 |
| ROC-AUC | 0.887 | **0.920** | **+0.033** |
| P@20 | 0.575 | 0.500 | **−0.075** |
| P@50 | 0.338 | 0.367 | +0.029 |

**Mixed result**: ROC-AUC 큰 폭 개선이지만 PR-AUC는 작은 증가, P@20 오히려
하락. 시장 피처가 ranking 전반에는 도움되지만 양성 top-K 정밀도에는 부정적.
가설: 코스닥 위주 데이터에서 시장 가격이 펀더멘털을 덜 정확히 반영. 거래
정지 빈번. Campbell+(2008)이 SP500에서 본 효과가 한국 코스닥에서는
부분적으로만 재현.

### 7.2 A1 — 감사의견 (Audit Opinion) ★

전체 보고서: `docs/research_a1_audit_features_report.md`.

**데이터**: DART OpenAPI `accnutAdtorNmNdAdtOpinion.json` 엔드포인트로
1,536 corp × 2014~2024 11년 = **16,896 호출**, 성공 0 실패. 적정/한정/
부적정/의견거절 텍스트 → 정수 인코딩.

**시점 누수 차단**: 사업보고서(감사보고서 포함)는 회계연도 다음 해 3월
공개. 따라서 (stock, year=Y, *)는 **Y-1 의견까지만 사용**. Y 의견은 미래
정보로 절대 미사용.

**5개 피처:**

| 피처 | 정의 |
|------|------|
| audit_opinion_t | Y-1 감사의견 (0=적정, 1=한정, 3=의견거절) |
| audit_opinion_t1 | Y-2 감사의견 |
| audit_nonclean_consec | Y-1부터 거꾸로 연속 비적정 횟수 (KOSDAQ 38조 트리거) |
| audit_nonclean_5y | Y-5~Y-1 비적정 누적 |
| audit_observed | Y-1 의견 데이터 존재 여부 (mask) |

**라벨 vs 감사의견 단순 교차표** (이론 그대로):

| audit_nonclean_consec | 음성 | 양성 | 양성 비율 | base rate 대비 |
|----------------------|------|------|-----------|---------------|
| 0회 | 31,228 | 126 | **0.40%** (base) | 1× |
| 1회 | 276 | 81 | 22.7% | **57배** |
| 2회 | 72 | 45 | **38.5%** | **96배** |
| 3회 | 11 | 8 | 42.1% | **105배** |

KRX 코스닥 상장규정 38조 "2년 연속 비적정 = 상폐 사유" 가 데이터로 직접
재현. Geiger & Raghunandan(2002) 미국 결과의 한국 강한 재현.

**결과 (Transformer walk-forward 평균):**

| 지표 | 33 feat | 39 (+ market) | **44 (+ market + audit)** | Δ vs 33 |
|------|---------|----------------|----------------------------|---------|
| Test PR-AUC | 0.289 | 0.297 | **0.408** | **+0.119** |
| ROC-AUC | 0.887 | 0.920 | 0.919 | +0.032 |
| P@20 | 0.575 | 0.500 | **0.650** | +0.075 |
| P@50 | 0.338 | 0.367 | **0.440** | +0.102 |

**4/4 fold 모두 +0.10 이상 개선** (fold 1 +0.131, fold 2 +0.135, fold 3
+0.110, fold 4 +0.101). 시장 피처가 P@20을 하락시켰던 것과 정반대 — 감사
피처는 양성 영역 정밀도를 직접 끌어올린다. **단일 fold 4 PR-AUC 0.444는
S0 baseline 95% CI 상단 0.421을 명확히 넘김**.

### 7.3 A2 — 관리종목 지정 이력 (Negative Marginal)

전체 보고서: `docs/research_a2_supervision_features_report.md`.

**데이터**: DART list API의 거래소공시(`pblntf_ty='I'`)에서 `report_nm`
키워드("관리종목", "투자주의환기", "투자경고", "투자위험", "단기과열")
필터링. 1,506 corp 호출, 655 이벤트, 0 실패.

**6개 피처**: is_supervised_now, days_since_last_concern,
n_supervision_events_5y, n_concern_events_5y, has_trading_halt_3y,
has_any_supervision_history.

**라벨 vs is_supervised_now**:

| is_supervised_now | 양성 비율 | base rate 대비 |
|-------------------|-----------|---------------|
| 0 | 0.46% | 1× |
| 1 (관리종목) | **5.67%** | **12배** |

의미 있는 신호이나 A1 감사의견 (96배) 대비 **약 8배 약함**.

**결과 (Transformer walk-forward 평균):**

| 지표 | 44 (+ market + audit) | **50 (+ supervision)** | Δ |
|------|------------------------|-------------------------|---|
| Test PR-AUC | 0.408 | 0.402 | **−0.006** |
| ROC-AUC | 0.919 | 0.920 | +0.001 |
| P@20 | 0.650 | 0.621 | −0.029 |
| P@50 | 0.440 | 0.408 | −0.032 |

**Negative result**: A2 추가의 효과가 사실상 0. **이유**: A1 감사의견과
A2 관리종목 지정은 KRX 코스닥 상장규정 28조~38조의 **같은 부실 트리거를
공유**한다. 비적정 감사의견이 관리종목 지정 사유의 일부이므로, A1을
이미 가진 상태에서 A2의 한계 효용은 거의 0.

이는 학술적으로 의미 있는 발견:
> 한국 코스닥에서 KRX 부실 표지 시스템의 두 정보원(감사의견, 관리종목)은
> 강하게 중복되며, 둘을 함께 사용해도 PR-AUC는 더 오르지 않는다.

### 7.4 외부 데이터 통합 결론

**현재 best baseline = Transformer + 시장 + 감사 (44 features)**. A2는
운영 모델에 포함하지 않고 보고서 가치 (negative result)만 남김.

---

## 8. 천장 돌파의 통계적 입증 (Best CI)

전체 보고서: `docs/research_best_ci_report.md`.

### 8.1 질문 재정의

walk-forward 평균 PR-AUC 0.408이 S0 baseline 95% CI [0.180, 0.421] 상단
**근처**에 도달했으나, 이것이 **통계적으로 명확한 돌파인지** 확정 못 했다.
walk-forward 평균은 4 fold 평균이라 baseline 단일 split과 직접 비교 불가.

### 8.2 방법 — Paired Bayesian Bootstrap

baseline과 **정확히 같은 조건**으로 비교:
- **단일 split** (train 2015~2022 / valid 2023 / test 2024)
- Best 모델 (Transformer + market + audit, 44 features) 5 seed 학습 →
  5-seed test probability 평균
- RF baseline 동일 5 seed 재학습 → 5-seed probability 평균
- 같은 Dirichlet(1,…,1) 가중치로 두 모델 PR-AUC posterior 동시 산출
  (paired n=5,000)
- Δ = best − baseline posterior

```
for i in 1..5000:
    w_i = Dirichlet(1, …, 1)     # n = 5,311
    best_post[i] = AP(y_test, best_proba_avg, w_i)
    base_post[i] = AP(y_test, rf_proba_avg, w_i)
```

paired가 핵심 — 두 모델의 차이가 진짜 신호인지 단순 모델 noise인지 직접
검증.

### 8.3 결과

**5-seed 분포:**

| seed | Best | Baseline |
|------|------|----------|
| 42 | 0.462 | 0.288 |
| 7 | 0.447 | 0.277 |
| 13 | 0.425 | 0.248 |
| 21 | 0.385 | 0.264 |
| 100 | 0.441 | 0.271 |
| **mean** | **0.432 ± 0.026** | **0.269 ± 0.013** |

**Posterior 비교:**

| 항목 | Best | Baseline |
|------|------|----------|
| Ensemble PR-AUC | 0.454 | 0.270 |
| 95% Bayesian CI | **[0.331, 0.586]** | [0.167, 0.402] |

**Paired Δ posterior:**

| 항목 | 값 |
|------|-----|
| Δ mean | **+0.183** |
| Δ 95% CI | **[+0.094, +0.279]** ← **전부 양수** |
| **P(best > baseline)** | **1.000** (5,000/5,000) |

### 8.4 통계적 결론

1. **Δ 95% CI 전부 양수** → 5% 양측 유의수준에서 best > baseline 입증.
2. **P(best > baseline) = 1.000** → 5,000번 paired bootstrap 중 baseline이
   best를 이긴 경우 0건.
3. **Best CI 하단 0.331 > baseline mean 0.270** → 두 posterior가 거의
   겹치지 않음.
4. **Δ 5% percentile = +0.094** → 95% 신뢰수준에서 **최소 +9.4%p 우위 보장**.

**S0 baseline의 95% CI [0.180, 0.421]을 완전 돌파.** 천장은 통계적으로
확실하게 깨졌다.

### 8.5 천장 돌파의 핵심 기여 분해

| 구성요소 | 누적 PR-AUC | 단계 +Δ |
|----------|-------------|---------|
| RF baseline (S0) | 0.270 | — |
| → 시계열 Transformer (Phase 3) | 0.342 | +0.072 |
| → + 시장 피처 (A3) | 0.297 (WF) | +0.008 |
| → + 감사의견 (A1) | **0.408 (WF) / 0.432 (single 5-seed)** | **+0.119 (WF)** / +0.163 (single) |
| → + 관리종목 (A2) | 0.402 (WF) | −0.006 (중복) |

**A1 감사의견이 단일 가장 큰 단일 기여 요소**. 시계열 모델링 + KRX 부실
신호 결합이 천장 돌파의 90%+를 설명한다.

### 8.6 한계 (Best CI)

- **단일 split** — fold 4 조건의 통계 입증이며, walk-forward 평균의 CI는
  별도 측정 필요.
- **5 seed로 std 추정** — 10+ seed로 검증 시 std 정밀화.
- **외부 valid 미사용** — 모든 통계는 in-sample. 2025 데이터 도착 시
  진정한 out-of-sample 검증 필요.

---

## 9. 종합 및 해석

### 9.1 본 연구의 narrative

```
[출발] baseline 0.2876 = 9개 실험 못 넘는 천장
         ↓ 진단 1 (S0)
[발견 1] 천장이 단단하지 않음. CI [0.18, 0.42] 폭 0.24.
         9개 실험 통계적으로 noise 안.
         ↓ 진단 2 (SR)
[발견 2] 라벨링 변경으로도 천장 안 깸. 7개 모두 CI 안.
         L3 rolling_H24가 가장 일관 → 후속 default.
         ↓ 시계열 (S1 Phase 1~3 + WF)
[발견 3] 단일 스냅샷 → 시계열 전환이 +0.076 PR-AUC.
         Transformer + Time2Vec이 새 baseline. 4/4 fold 우위.
         ↓ 외부 데이터 (A3, A1, A2)
[발견 4] 감사의견(A1) +0.119 — Geiger 2002 한국 재현.
         시장(A3) 작음. 관리종목(A2) A1과 중복.
         ↓ 통계 검증 (Best CI)
[종결] Δ CI [+0.094, +0.279] 전부 양수, P = 1.000.
       천장 통계적으로 명확히 깨짐.
```

### 9.2 학술적 의미

1. **Diagnostic-first methodology의 가치**: 9개 사전 실험의 "실패"를
   정직하게 noise로 재해석함으로써 다음 연구 방향(시계열+외부)을 정확히
   가리킬 수 있었다.

2. **Geiger 2002의 한국 재현**: 미국 데이터 기반 감사의견 부도 예측력
   이론이 한국 코스닥에서 강하게 재현됨을 96배 lift로 정량화. 향후 한국
   금융 ML 연구에서 감사 시그널의 활용도 확립.

3. **Campbell 2008의 한국 부분 재현**: 시장 변수의 부도 예측력은 미국
   SP500 기반 결과 대비 약함. 코스닥 소형주의 가격 noise 특성이 원인으로
   추정됨. 한국 시장 ML 연구에 시사점.

4. **이상탐지 framing의 검증**: 비지도 5종(IsolationForest/LOF/OCSVM/HBOS/
   ECOD)이 supervised RF에 0.13~0.25 PR-AUC 뒤짐. 본 문제는 supervised
   label의 정보가 본질적 — pure anomaly detection으로 가지 말아야 함.

5. **방법론적 contribution**: Bayesian bootstrap, paired permutation, MMD
   shift detection, walk-forward CV의 통합 적용이 본 분야 한국 연구에서는
   드물다.

### 9.3 운영적 의미

- top-20 양성 적중률 0.65: 가장 위험한 20개 기업 중 13개 적중 (baseline의
  11~12개 대비 향상).
- ECE 0.003 (매우 잘 보정) → 확률 출력이 의미 있는 위험 점수로 사용 가능.
- 시점 누수 차단 설계 → 운영 시점에 실시간 사용 가능.

---

## 10. 한계와 후속 연구

### 10.1 본 연구의 한계

| 한계 | 영향 | 보완 방향 |
|------|------|-----------|
| **단일 test 연도 (2024)** | 진정한 OoS 검증 부재 | 2025 데이터 도착 시 검증 |
| **K=3 고정** | 더 긴 시계열 효과 미확인 | K=2,4,5 sensitivity |
| **3 seed walk-forward** | std 추정 약함 | 5+ seed 확대 |
| **Walk-forward 평균의 통계 CI** | 단일 split CI만 산출 | 평균에 대한 bootstrap |
| **Hyperparameter 단일 setting** | 추가 향상 가능성 | Sweep (model_dim, dropout) |
| **재무 데이터 결측** | imputed 데이터로 학습 (NaN ratio 진단에서 신호 약함 확인) | feature mask channel은 신호 작음 |
| **외부 valid 부재** | in-sample 통계만 | 2025 도착 후 |

### 10.2 후속 stream 권장

| 항목 | 기대 효과 | 비용 | 우선순위 |
|------|-----------|------|---------|
| Walk-forward × 5+ seed bootstrap | 평균의 CI 확립 | 2~3시간 | 권장 |
| Hyperparameter sweep | +0.01~0.03 | 6~8시간 | 권장 |
| **Stream 3 hazard 모형** (Cox PH, DeepSurv) | fixed-N 의존성 제거, 학술적 차별성 | 1~2주 | **강력 권장** |
| **Phase 4 Neural CDE** (Kidger 2020) | UNIST DAL 시그니처 직결, K 임의 길이 | 1주 + GPU | **권장 (GPU 확보 시)** |
| VAE reconstruction error를 feature로 | hybrid anomaly framing | 1주 | 선택 |
| 2025 데이터 OoS 검증 | 진정한 일반화 검증 | 2025 데이터 대기 | 필수 |

### 10.3 UNIST DAL 연결 지점

| 랩 시그니처 | 본 연구 매칭 |
|-------------|--------------|
| Bayesian bootstrap monitoring statistic (Rubin 1981) | S0 진단, Best CI 모두 핵심 도구로 사용 |
| Irregular time-series classification (Neural CDE, FlowPath) | Q1/H1/Q3/ANNUAL 혼재 시계열을 Time2Vec + Transformer로 처리, Phase 4에 NCDE 계획 |
| 산업 이상탐지 (AIS, 원전, 제조) | 상폐 = 재무 시계열 이상탐지로 framing, 비지도 5종 vs supervised 비교 |
| 통계 엄밀성 (paired test, permutation, MMD) | 모든 비교를 paired bootstrap + Holm 보정 |

향후 본 stream의 자연스러운 확장 방향:
- Phase 4 NCDE: 분기·반기 혼재 시계열 그대로 처리 (현재 동일 quarter
  타입 분리 방식의 한계 극복)
- VAE 기반 monitoring statistic: 재무 시계열의 reconstruction error를
  추가 anomaly 신호로
- 베이지안 부트스트랩으로 운영 임계치 분포 추정 → 위험 점수 변동 보고

---

## 11. 재현성 (Reproducibility)

### 11.1 환경

- Python 3.13.2
- PyTorch 2.12 CPU (수동 설치)
- scikit-learn 1.8, scipy 1.17, pandas 3.0, matplotlib 3.10
- pyod, shap, FinanceDataReader, dart-fss, dotenv
- DART OpenAPI 키 (.env)

### 11.2 전체 재현 명령 시퀀스

```powershell
# === S0 진단 ===
.venv\Scripts\python.exe -m src.research.s0_diagnostic.run_s0_A1_bootstrap
.venv\Scripts\python.exe -m src.research.s0_diagnostic.run_s0_A2_walkforward
.venv\Scripts\python.exe -m src.research.s0_diagnostic.run_s0_A3_paired_tests
.venv\Scripts\python.exe -m src.research.s0_diagnostic.run_s0_A4_dist_shift
.venv\Scripts\python.exe -m src.research.s0_diagnostic.run_s0_A5_unsupervised
.venv\Scripts\python.exe -m src.research.s0_diagnostic.run_s0_A6_interpretation
.venv\Scripts\python.exe -m src.research.s0_diagnostic.aggregate

# === SR 라벨링 ===
.venv\Scripts\python.exe -m src.research.sr_labeling.run_sr_all

# === S1 시계열 (Phase 1~3) ===
.venv\Scripts\python.exe -m src.research.s1_irregular_ts.run_s1_phase1
.venv\Scripts\python.exe -m src.research.s1_irregular_ts.run_s1_phase2
.venv\Scripts\python.exe -m src.research.s1_irregular_ts.run_s1_phase3

# === S1 walk-forward (33 features) ===
.venv\Scripts\python.exe -m src.research.s1_irregular_ts.run_s1_walkforward `
    --seeds 42,7,13 --epochs 25

# === 외부 데이터 (수집 + 피처) ===
.venv\Scripts\python.exe -m src.research.a3_market.fetch_ohlcv --workers 6
.venv\Scripts\python.exe -m src.research.a3_market.market_features
.venv\Scripts\python.exe -m src.research.a1_audit.fetch_audit --workers 4
.venv\Scripts\python.exe -m src.research.a1_audit.audit_features
.venv\Scripts\python.exe -m src.research.a2_supervision.fetch_supervision --workers 4
.venv\Scripts\python.exe -m src.research.a2_supervision.supervision_features

# === walk-forward (외부 데이터 포함) ===
.venv\Scripts\python.exe -m src.research.s1_irregular_ts.run_s1_walkforward `
    --seeds 42,7,13 --epochs 25 --with-market --with-audit
.venv\Scripts\python.exe -m src.research.s1_irregular_ts.run_s1_walkforward `
    --seeds 42,7,13 --epochs 25 --with-market --with-audit --with-supervision

# === Best CI 통계 검정 ===
.venv\Scripts\python.exe -m src.research.best_ci.run_best_bootstrap

# === 보조 진단 (NaN ratio) ===
.venv\Scripts\python.exe -m src.research.nan_diag.diagnose_nan_signal

# === Portfolio Hero figure ===
.venv\Scripts\python.exe -m src.research.portfolio.hero_figure
```

전체 wall time ≈ **5~6시간** (CPU, GPU 불필요). 모든 seed/bootstrap 고정,
동일 seed 재실행 시 결과 byte 단위 일치.

### 11.3 디스크 사용량

| 디렉터리 | 크기 (대략) |
|----------|------------|
| `data/market_ohlcv/` | ~200 MB (1,516 OHLCV CSV) |
| `data/audit_opinion/` | ~50 MB (16,896 JSON) |
| `data/supervision/` | ~5 MB (1,536 JSON) |
| `preprocess/data/` (전처리) | ~250 MB |
| `results/` (모든 stream) | ~30 MB |
| `docs/` (보고서) | ~2 MB |

---

## 12. 참고 문헌

### 통계 방법론
- Rubin, D.B. (1981). *The Bayesian Bootstrap.* Annals of Statistics, 9(1).
- Sun, X. & Xu, W. (2014). *Fast implementation of DeLong's algorithm.* IEEE
  Signal Processing Letters.
- Gretton, A. et al. (2012). *A kernel two-sample test.* JMLR, 13.
- Guo, C. et al. (2017). *On calibration of modern neural networks.* ICML.

### 부실 예측 — 회계
- Beaver, W.H. (1966). *Financial Ratios as Predictors of Failure.* JAR.
- Altman, E.I. (1968). *Financial Ratios, Discriminant Analysis, and the
  Prediction of Corporate Bankruptcy.* Journal of Finance.
- Ohlson, J.A. (1980). *Financial Ratios and the Probabilistic Prediction of
  Bankruptcy.* JAR.
- Tinoco, M.H. & Wilson, N. (2013). *Financial Distress and Bankruptcy
  Prediction among Listed Companies.* IJFE.

### 부실 예측 — 시장
- Shumway, T. (2001). *Forecasting Bankruptcy More Accurately: A Simple
  Hazard Model.* Journal of Business.
- Campbell, J.Y., Hilscher, J. & Szilagyi, J. (2008). *In Search of Distress
  Risk.* Journal of Finance.
- Bharath, S.T. & Shumway, T. (2008). *Forecasting Default with the Merton
  Distance to Default Model.* RFS.

### 부실 예측 — 감사·지배구조
- Geiger, M.A. & Raghunandan, K. (2002). *Going-Concern Opinions and the
  Prediction of Corporate Failure.* Auditing: A Journal of Practice & Theory.
- Lennox, C. (1999). *Identifying Failing Companies: A Reevaluation of the
  Logit, Probit, and DA Approaches.* J. Accounting & Economics.

### 생존 분석
- Cox, D.R. (1972). *Regression Models and Life Tables.* JRSSB.
- Lane, W.R., Looney, S.W. & Wansley, J.W. (1986). *An Application of the
  Cox Proportional Hazards Model to Bank Failure.* JBF.
- Katzman, J. et al. (2018). *DeepSurv: Personalized Treatment Recommender
  System Using a Cox Proportional Hazards Deep Neural Network.* BMC Medical
  Research.

### 시계열 딥러닝
- Cho, K. et al. (2014). *Learning Phrase Representations using RNN
  Encoder-Decoder for Statistical Machine Translation.* EMNLP.
- Che, Z. et al. (2018). *Recurrent Neural Networks for Multivariate Time
  Series with Missing Values.* Scientific Reports.
- Vaswani, A. et al. (2017). *Attention Is All You Need.* NeurIPS.
- Kazemi, S.M. et al. (2019). *Time2Vec: Learning a Vector Representation of
  Time.* arXiv:1907.05321.
- Kidger, P. et al. (2020). *Neural Controlled Differential Equations for
  Irregular Time Series.* NeurIPS.
- Rubanova, Y. et al. (2019). *Latent ODEs for Irregularly-Sampled Time
  Series.* NeurIPS.

### 이상탐지
- Kingma, D.P. & Welling, M. (2014). *Auto-Encoding Variational Bayes.* ICLR.
- Park, D. et al. (2018). *A Multimodal Anomaly Detector for Robot-Assisted
  Feeding Using an LSTM-Based VAE.* IEEE RA-L.
- An, J. & Cho, S. (2015). *Variational Autoencoder Based Anomaly Detection
  Using Reconstruction Probability.* SNU Tech Report.
- Liu, F.T., Ting, K.M., Zhou, Z.-H. (2008). *Isolation Forest.* ICDM.
- Breunig, M.M. et al. (2000). *LOF: Identifying Density-Based Local Outliers.*
  SIGMOD.

### 클래스 불균형
- Lin, T.-Y. et al. (2017). *Focal Loss for Dense Object Detection.* ICCV.

### 한국 규정
- 한국거래소(KRX) 코스닥시장 상장규정 제28조~제38조 (관리종목 지정 및
  상장폐지 사유).
- DART OpenAPI 정기보고서 주요정보 (감사인의 명칭 및 감사의견).

---

## 13. 부록 — 코드 구조 및 산출물 목록

### 13.1 코드 디렉터리

```
src/research/
├── s0_diagnostic/          # S0 6개 진단 실험
│   ├── baseline.py
│   ├── walk_forward_cv.py
│   ├── bayesian_bootstrap.py
│   ├── paired_tests.py
│   ├── distribution_shift.py
│   ├── unsupervised_baselines.py
│   ├── interpretation.py
│   ├── aggregate.py
│   └── run_s0_A1_bootstrap.py ~ run_s0_A6_interpretation.py
├── sr_labeling/            # SR 7개 라벨링 비교
│   ├── labelings.py
│   ├── train_eval.py
│   └── run_sr_all.py
├── s1_irregular_ts/        # S1 시계열 (Phase 1~3 + walk-forward)
│   ├── sequences.py
│   ├── sequences_grud.py
│   ├── gru_baseline.py
│   ├── gru_d.py
│   ├── transformer_t2v.py
│   ├── run_s1_phase1.py
│   ├── run_s1_phase2.py
│   ├── run_s1_phase3.py
│   └── run_s1_walkforward.py
├── a3_market/              # 시장 OHLCV + 분기 피처
│   ├── fetch_ohlcv.py
│   └── market_features.py
├── a1_audit/               # 감사의견
│   ├── corp_map.py
│   ├── fetch_audit.py
│   └── audit_features.py
├── a2_supervision/         # 관리종목 공시
│   ├── fetch_supervision.py
│   └── supervision_features.py
├── nan_diag/               # NaN ratio 진단
│   └── diagnose_nan_signal.py
├── best_ci/                # Best baseline 통계 검정
│   └── run_best_bootstrap.py
└── portfolio/              # Portfolio 산출물
    └── hero_figure.py
```

### 13.2 산출물 디렉터리

```
results/
├── research_s0_diagnostic/         # A1~A6 + aggregate
├── research_sr_labeling/           # 7개 라벨링 비교
├── research_s1_phase1/             # GRU
├── research_s1_phase2/             # GRU-D
├── research_s1_phase3/             # Transformer
├── research_s1_walkforward/        # 4 fold × 3 seed (33 feat)
├── research_s1_walkforward_market/                       # 39 feat
├── research_s1_walkforward_market_audit/                 # 44 feat
├── research_s1_walkforward_market_audit_supervision/     # 50 feat
├── research_nan_diag/              # NaN ratio 진단
└── research_best_ci/               # Best CI 통계 검정
```

### 13.3 보고서 (`docs/`)

- `research_s0_diagnostic_report.md` — S0 진단 6 실험
- `research_sr_labeling_report.md` — SR 라벨링 점검
- `research_s1_phase1_report.md` — GRU baseline
- `research_s1_phase2_report.md` — GRU-D (negative)
- `research_s1_phase3_report.md` — Transformer (new best)
- `research_s1_walkforward_report.md` — Walk-forward 재검증
- `research_a3_market_features_report.md` — 시장 피처
- `research_a1_audit_features_report.md` — 감사의견 (key breakthrough)
- `research_a2_supervision_features_report.md` — 관리종목 (redundant)
- `research_best_ci_report.md` — 천장 돌파 통계 검정
- `다음 실험 설계안(연구지향 Stream 0).md` — 초기 설계안

### 13.4 Portfolio 산출물 (`docs/portfolio/`)

- `portfolio_one_pager_en.md` — 영문 1-pager (지원서/이메일 첨부용)
- `portfolio_technical_report_ko.md` — 본 문서
- `hero_figure.png` — narrative 한 장 시각화

---

*이 문서는 학부 portfolio 통합본으로, 모든 수치·결과·인용은 위 보고서 시리즈와 코드 저장소에서 재현 가능하다. 대학원 지원·면접·자기소개서·자기 정리 용으로 자유 사용.*
