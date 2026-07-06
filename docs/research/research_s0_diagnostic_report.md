# Baseline PR-AUC 0.2876의 천장은 얼마나 견고한가?
## 한국 상장폐지 예측 baseline의 진단적 연구

작성일: 2026-05-28
작성자: 이현지
관련 계획: `docs/다음 실험 설계안(연구지향 Stream 0).md`
재현 코드: `src/research/s0_diagnostic/`
결과 산출물: `results/research_s0_diagnostic/`

---

## 초록 (Abstract)

본 보고서는 한국 상장폐지 조기경보 모델의 보고된 PR-AUC 천장값 0.2876을
진단적 관점에서 재평가한다. 대상 모델은 분기 단위 DART 재무비율을 활용한
Random Forest baseline (33 피처 = 27 재무비율 + 3 YoY 증가율 + 6 거시경제 +
`signed_log1p` 변환, fixed_N1 라벨, train 2015~2022 / valid 2023 / test 2024)
이며, 그간 시도된 9개 개선 실험(`exp_010`~`exp_018`: RF/XGB/LGBM Optuna,
피처 엔지니어링, class_weight, SMOTE, Piotroski F-score, fixed_N2 라벨)은
단일 test 2024에서 모두 baseline 값을 넘지 못했다.

본 진단에서는 (i) Bayesian / Frequentist stratified bootstrap으로 test set에
대한 baseline의 신뢰구간을 추정하고, (ii) 4-fold walk-forward CV × 5 seed
= 20개 측정으로 평가 안정성을 분석하며, (iii) baseline posterior 기반
Bayesian 비교 + Holm-Bonferroni 보정으로 exp_010~018 각각을 baseline과 통계적
으로 비교한다. 또한 (iv) PSI / KL / JS divergence와 permutation 검정을 곁들인
다변량 MMD로 분포 이동을 정량화하고, (v) IsolationForest, LOF, OCSVM, HBOS,
ECOD의 5가지 비지도 이상탐지 baseline을 supervised RF와 비교하며, (vi) 다중
seed permutation importance, TreeSHAP, Expected Calibration Error로 모델
해석과 보정 상태를 점검한다.

세 가지 핵심 발견이 다음 단계 연구 방향을 재조정한다.
**(1)** 0.2876은 seed 42의 최댓값이며, test PR-AUC의 95% Bayesian 신뢰구간
은 **[0.180, 0.421]** — 폭이 0.24에 달한다. 5개 seed 평균은 **0.270 ± 0.015**
로 떨어진다.
**(2)** 9개 모든 사전 실험이 baseline의 95% CI 안에 들어가며, Holm 보정 후
유의수준 0.05에서 baseline과 통계적으로 구분되는 실험은 0개다.
**(3)** train~2022와 test 2024 사이의 분포 이동은 연도-keyed 거시변수가
지배한다. 재무비율 자체는 대부분 안정적이며 (부채비율 한 개만 예외), supervised
RF는 모든 비지도 이상탐지를 0.13~0.25 PR-AUC만큼 능가한다. 이는 라벨에
유의미한 정보가 있다는 뜻이며, **순수 anomaly detection** 프레이밍만으로는
이 정보를 복원할 수 없음을 의미한다.

결론적으로 "0.2876 천장"은 모델의 한계가 아니라 **평가 자체의 노이즈가
일부 기여하는 현상**임이 확인된다. 본 진단 결과를 토대로 후속 연구 stream
(불규칙 시계열 모델링, VAE 보조 피처, 생존 분석식 재라벨링)의 우선순위를
재조정한다.

---

## 1. 서론 (Introduction)

본 프로젝트의 baseline은 33 피처(27 재무비율 + 3 YoY 증가율 + 6 거시경제
지표, `signed_log1p` 스케일링 적용)를 사용하는 단일 스냅샷 Random Forest다.
연간 데이터로 학습(2015~2022), 검증(2023, 양성 39), 평가(2024, 양성 56)하며
라벨은 `fixed_N1`(상폐 정확히 1년 전 데이터를 양성으로 표시). 124:1의 극심한
클래스 불균형과 약 5,000행의 test 셋이라는 조건에서 test PR-AUC가 사실상
모델 비교의 헤드라인이다.

지금까지 시도된 9개 개선 실험은 모두 baseline의 PR-AUC 0.2876을 넘지 못했다
(`docs/5 20 모델 개선 실험 결과보고서(exp_010~018).md`). 이 사실로부터 자연스
럽게 떠오르는 질문은: *0.2876이 정말 천장인가, 아니면 그저 표본 추출 분포의
한 점에 불과한가?*

본 보고서는 `docs/다음 실험 설계안(연구지향 Stream 0).md`의 Stream 0를
실행한 결과다. 6개 진단 실험을 통해 (a) baseline의 통계적 불확실성을 측정
하고, (b) 이전 실패가 noise였는지 signal이었는지를 검증하며, (c) 어느 후속
stream에 투자할 가치가 있는지 판단할 근거를 모은다.

---

## 2. Baseline의 신뢰구간 (A1)

`exp_018`의 정확한 설정을 재현했다 (RF: n_estimators=200, max_depth=10,
min_samples_leaf=5, max_features=sqrt, random_state=42). 보고된 수치를 정확히
복원: **test PR-AUC = 0.2876**, ROC-AUC = 0.8586.

이어서 test 셋에 대해 Bayesian bootstrap (Dirichlet(1,…,1) 가중,
Rubin 1981) 2000회와 Stratified frequentist bootstrap (양성·음성 비율 보존)
2000회를 각각 실행했다.

| 지표    | 방법         | 평균    | 중앙값  | 95% CI            | 표준편차 |
|---------|--------------|---------|---------|--------------------|----------|
| PR-AUC  | Bayesian     | 0.2938  | 0.2887  | **[0.180, 0.421]** | 0.062    |
| PR-AUC  | Frequentist  | 0.2915  | 0.2879  | [0.177, 0.412]     | 0.061    |
| ROC-AUC | Bayesian     | 0.8589  | 0.8589  | [0.807, 0.902]     | 0.025    |
| ROC-AUC | Frequentist  | 0.8586  | 0.8586  | [0.803, 0.904]     | 0.025    |

PR-AUC의 95% 신뢰구간 폭은 0.24. 양성 표본이 56개에 불과하므로 PR-AUC의
표준오차는 약 0.06 — 이는 baseline과 이전 실험들의 차이와 같은 자릿수다.
ROC-AUC는 절대 ranking에 의존하므로 신뢰구간 폭이 0.10으로 훨씬 좁다.

> 사후분포 밀도 그림: `results/research_s0_diagnostic/A1_posterior_density.png`

**RQ1에 대한 답**: 0.2876은 매우 정밀한 점추정이지만 그 기반이 되는
사후분포는 0.18~0.42를 모두 포함한다.

---

## 3. Walk-Forward CV × Multi-Seed (A2)

기존의 단일 train(2015~2022) / valid(2023) / test(2024) 분할을 연도 cutoff로
확장하여 4개 walk-forward fold를 구성하고, 각 fold에서 5개 seed (42, 7, 13,
21, 100)로 baseline RF를 학습·평가했다. 동일하게 test 2024에 대해서도
5개 seed로 평가.

| Fold | Train 연도   | Valid 연도 | Valid 양성 | Valid PR-AUC mean ± std |
|------|---------------|------------|-------------|--------------------------|
| 1    | 2015~2019     | 2020       | 69          | 0.1924 ± 0.0041          |
| 2    | 2015~2020     | 2021       | 53          | 0.2390 ± 0.0076          |
| 3    | 2015~2021     | 2022       | 23          | 0.2387 ± 0.0391          |
| 4    | 2015~2022     | 2023       | 39          | **0.1163 ± 0.0054**      |

| Seed | Test 2024 PR-AUC | ROC-AUC |
|------|------------------|---------|
| 42   | **0.2876**       | 0.8586  |
| 7    | 0.2767           | 0.8513  |
| 13   | 0.2483           | 0.8553  |
| 21   | 0.2639           | 0.8560  |
| 100  | 0.2708           | 0.8560  |
| 평균 | **0.2695 ± 0.0147** | 0.8555 |

두 가지 사실이 중요하다.

1. **2023 valid는 비정상적으로 낮다** (PR-AUC 0.116). 2020, 2021, 2022 valid
   는 0.19~0.24, test 2024는 0.27 수준인데 2023만 0.12 부근이다. 즉 기존
   설정에서 valid로 선택한 연도가 우연히도 가장 어려운 연도였다. 이는
   hyperparameter 선택의 신뢰성에 큰 의문을 던진다.
2. **0.2876은 seed 최댓값**이지 평균이 아니다. 5개 seed 평균은 0.270,
   범위는 약 0.040. 따라서 단일 seed 기준 baseline-실험 비교는 ±0.02 정도의
   seed noise를 본질적으로 포함한다.

> Fold × seed heatmap: `results/research_s0_diagnostic/A2_fold_seed_grid.png`

**RQ1 보강**: 평가 셋을 고정해도 random seed만 바꿔도 test PR-AUC가
이전 실험들의 차이와 같은 규모로 변동한다.

---

## 4. exp_010~018 vs Baseline 통계적 비교 (A3)

9개 사전 실험 각각에서 보고된 test PR-AUC의 최고값(`best variant`)을 추출하여
A1의 baseline posterior와 비교했다.

| 실험                    | Best variant   | Test PR-AUC | Δ vs 0.2876 | 95% CI 안 | Holm 보정 p |
|-------------------------|----------------|-------------|-------------|------------|-------------|
| exp_010_threshold       | F1 strategy    | 0.2876      | +0.0000     | 예         | 1.000       |
| exp_011_optuna_rf       | baseline ref   | 0.2876      | +0.0000     | 예         | 1.000       |
| exp_012_optuna_xgb      | tuned_f1       | 0.2078      | −0.0798     | 예         | 1.000       |
| exp_013_fe              | Step0_baseline | 0.2876      | +0.0000     | 예         | 1.000       |
| exp_014_class_weight    | None / f1thr   | 0.2876      | +0.0000     | 예         | 1.000       |
| exp_015_optuna_lgbm     | tuned_f1       | 0.1930      | −0.0946     | 예         | 1.000       |
| exp_016_smote           | None / f1      | 0.2876      | +0.0000     | 예         | 1.000       |
| exp_017_piotroski       | baseline / f1  | 0.2876      | +0.0000     | 예         | 1.000       |
| exp_018_fixed_n         | N1 / exp-A     | 0.2876      | +0.0000     | 예         | 1.000       |

양측 p-value는 `2 · min(P(baseline ≥ exp), P(baseline ≤ exp))`로 baseline
posterior 위에서 계산했고, 9개 비교에 대해 Holm-Bonferroni 보정을 적용했다.

**RQ2에 대한 답**: 9개 사전 실험 그 어느 것도 baseline과 통계적으로 구분
되지 않는다. 가장 시각적으로 차이가 커 보였던 exp_012 (XGB Optuna, 0.208)와
exp_015 (LGBM Optuna, 0.193)도 보정 전 p-값이 각각 0.156, 0.087로 0.180~0.421
신뢰구간 안에 완전히 흡수된다. 다시 말해 "성능이 떨어진 실험"은 더 못한
모델이 아니라 **같은 분포에서 추출된 무작위 표본**이었다.

> 그림: `results/research_s0_diagnostic/A3_pr_auc_vs_baseline_ci.png`

이는 exp_010~018의 음성 결과(negative result)를 재해석하게 한다. 그것들은
개입 자체의 실패가 아니라 **평가 셋이 ±0.05~0.10 PR-AUC 규모의 효과를
탐지하기에 너무 작고 좁다**는 증거다.

---

## 5. 분포 이동 정량화 (A4)

train (2015~2022) / valid (2023) / test (2024) 사이의 공변량 이동(covariate
shift)을 세 단계로 측정했다.

### 5.1 피처별 단변량 divergence

PSI(train→test) > 0.25(전통적 "significant shift" 기준)을 넘는 피처는 33개
중 **7개**다. 이 중 **6개는 거시경제 변수**다 — 거시변수는 한 연도에 하나의
값을 가지므로 train(8년 분량)과 test(1년 분량)을 비교하면 구조적으로 PSI가
폭발한다.

| 피처               | PSI(train→valid) | PSI(train→test) | PSI(valid→test) |
|--------------------|------------------|------------------|------------------|
| vix_avg            | 9.15             | 10.57            | 10.40            |
| cpi_yoy            | 12.45            | 10.14            | 2.49             |
| usdkrw_chg         | 9.00             | 9.20             | 2.49             |
| credit_spread      | 12.47            | 7.98             | 2.49             |
| kosdaq_return      | 8.75             | 7.96             | 10.25            |
| gdp_growth_yoy     | 10.68            | 7.92             | 6.55             |
| 부채비율           | 0.19             | **1.45**         | 0.60             |
| 비유동자산장기적합률 | 0.14             | 0.14             | 0.005            |
| 차입금의존도       | 0.03             | 0.06             | 0.006            |

거시 외의 유일한 outlier는 **부채비율** (PSI 1.45). 나머지 모든 재무비율은
0.10 이하다. *재무비율 분포 자체는 문제가 아니다.*

### 5.2 다변량 MMD (permutation test)

`signed_log1p` 스케일링 후 RBF 커널 (median bandwidth)에서 150회 permutation:

| 비교             | MMD²    | MMD     | 순열 p     | bandwidth σ |
|------------------|---------|---------|------------|-------------|
| train vs valid   | 0.00487 | 0.0698  | 0.000      | 15.05       |
| train vs test    | 0.01618 | 0.1272  | 0.000      | 14.84       |
| valid vs test    | 0.01674 | 0.1294  | 0.000      | 14.34       |

세 비교 모두 매우 유의하게 다른 분포다. 특히 **valid→test MMD가
train→valid MMD보다 크다** — 즉 2023과 2024가 서로 다른 정도가 2015~2022와
2023이 다른 정도보다 크다. 이는 "valid는 test의 적절한 예고편이다"라는
암묵적 가정과 정면으로 충돌하며, hyperparameter 선택이 valid에서 test로
완벽히 전이되지 못하는 이유를 정량적으로 설명한다.

### 5.3 섹터별 이동

(valid 2023, test 2024) 쌍을 GICS 섹터별로 분리하여 MMD 측정:

| 섹터                  | MMD²   | p     |
|------------------------|--------|-------|
| Consumer Staples       | 0.0267 | 0.025 |
| Consumer Discretionary | 0.0244 | 0.000 |
| Information Technology | 0.0203 | 0.000 |
| Communication Services | 0.0169 | 0.013 |
| Materials              | 0.0150 | 0.000 |
| Industrials            | 0.0149 | 0.000 |
| Health Care            | 0.0138 | 0.000 |

모든 섹터에서 유의한 2023→2024 drift가 확인된다. Consumer Staples가 가장
크고 Health Care가 가장 작다. 사전 실험 중 어느 것도 분포 이동 보정을
시도하지 않았다.

> 그림:
> `results/research_s0_diagnostic/A4_psi_top_features.png`
> `results/research_s0_diagnostic/A4_mmd_per_sector.png`

**RQ3에 대한 답**: 측정 가능한 분포 이동은 거시변수(연도 키)가 지배한다.
정작 다음 §7의 permutation 검정에서 예측력이 가장 낮게 나오는 피처들이다.
"신호의 본체"인 재무비율은 2015~2024 동안 대체로 안정적이며, 부채비율이
유일한 예외다.

---

## 6. 비지도 이상탐지 baseline (A5)

train 중 정상 기업(label=0)만으로 학습 (35,210 정상 기업·연도), test 2024에
대해 평가했다 (`pyod` 라이브러리).

| 모델                  | Test PR-AUC | 95% CI (Bayesian) | Δ vs supervised | P(sup ≥ this) |
|-----------------------|-------------|--------------------|------------------|---------------|
| RF (supervised, 기준) | **0.288**   | [0.180, 0.421]     | 0.000            | 0.500         |
| HBOS                  | 0.160       | [0.088, 0.278]     | −0.127           | 0.993         |
| IsolationForest       | 0.109       | [0.062, 0.213]     | −0.179           | 1.000         |
| OCSVM (subsampled)    | 0.086       | [0.046, 0.178]     | −0.201           | 1.000         |
| LOF                   | 0.036       | [0.015, 0.113]     | −0.251           | 1.000         |

ECOD는 Windows-Python 한글 경로 + joblib 호환 문제로 학습에 실패하여 제외
했다. 나머지 4개 모델의 결론은 일관된다.

**RQ4에 대한 답**: supervised RF는 모든 비지도 모델을 0.13~0.25 PR-AUC만큼
능가하며, supervised posterior의 99% 이상이 비지도 점추정보다 위에 위치한다.
**라벨에는 비지도 방법이 피처 분포만으로 복원할 수 없는 정보**가 들어있다.

이는 Stream 2의 anomaly detection 프레이밍을 정밀화한다 — VAE 같은 모델이
supervised RF를 **대체**하기는 어렵지만, 그 reconstruction error는 supervised
모델에 **보조 피처(feature)**로 투입될 가치는 있다. (§9.2)

> 그림:
> `results/research_s0_diagnostic/A5_unsupervised_baselines.png`
> `results/research_s0_diagnostic/A5_score_distribution.png`

---

## 7. 모델 해석 & Calibration (A6)

### 7.1 피처 중요도

3 seed × 5 repeat의 permutation importance (PR-AUC drop)와 TreeSHAP
global mean |·|는 대체로 일치하지만 최상위 한 자리에서 미묘하게 다르다.

| 순위 | Permutation (importance_mean) | TreeSHAP (mean) |
|------|-------------------------------|---------------------|
| 1    | 현금비율 (0.0395)              | 유보액/납입자본비율 (0.00285) |
| 2    | 순운전자본비율 (0.0172)        | 매출채권회전율 (0.00200)       |
| 3    | 매출원가율 (0.0120)            | 순운전자본비율 (0.00200)       |
| 4    | 매출채권회전율 (0.0110)        | 총자산증가율 (0.00196)         |
| 5    | 당좌비율 (0.0098)              | 자기자본비율 (0.00158)         |
| 6    | 매출총이익률 (0.0090)          | 매출액순이익률 (0.00154)       |
| 7    | 유형자산 (0.0087)              | 매출원가율 (0.00142)           |
| 8    | 유보액/납입자본비율 (0.0082)   | 현금비율 (0.00132)             |
| 9    | 총자본영업이익률 (0.0074)      | 매출총이익률 (0.00128)         |
| 10   | 차입금의존도 (0.0055)          | 총자본영업이익률 (0.00127)     |

두 ranking 모두 **유동성 지표(현금비율, 순운전자본비율, 당좌비율,
매출채권회전율)**, **수익성 지표(매출원가율, 매출총이익률, 총자본영업이익률,
매출액순이익률)**, **자본구조 지표(유보액/납입자본비율, 차입금의존도,
자기자본비율)**가 지배한다. 거시변수는 두 ranking 모두에서 한참 아래에
위치한다 (vix_avg는 permutation 13위, importance 0.002 — 현금비율의 1/20).

이는 §5와 정확히 일치한다 — 거시변수는 분포 이동은 크지만 신호는 미약하다.

> SHAP 그림: `results/research_s0_diagnostic/A6_shap_summary.png`

### 7.2 Calibration

| 지표         | 값      |
|--------------|---------|
| ECE (10 bin) | **0.0026** |
| Brier score  | **0.0092** |

모델은 매우 잘 보정되어 있다. ECE 0.003은 예측확률의 10분위 각각에서 평균
적으로 예측·관측 양성률 차이가 0.3%p에 불과함을 의미한다. 124:1 문제에서
이 사실은 중요하다 — **확률 ranking은 신뢰할 수 있다**. 임계치 기반의
F1/precision/recall이 불안정한 것은 양성 절대수가 적기 때문이지 확률 출력
자체의 질 때문이 아니다.

> 신뢰도 도표: `results/research_s0_diagnostic/A6_calibration_plot.png`

### 7.3 운영 임계치에서의 confusion

`exp_018`에서 사용된 F1 최적 임계치 0.097:

| TP | FP | FN | TN   | Recall | Precision |
|----|----|----|------|--------|-----------|
| 21 | 42 | 35 | 5213 | 37.5%  | 33.3%     |

35개의 FN과 42개의 FP는 `results/research_s0_diagnostic/A6_fp_fn_cases.csv`
에 메타 정보(섹터·연도·확률)와 함께 저장하여 후속 질적 검토를 위해 보존했다.

**RQ5에 대한 답**: baseline은 유동성·수익성·자본구조 지표라는 이론적으로
일관된 조합에 의존하며 거시변수는 거의 사용하지 않는다. 확률 출력은 잘
보정되어 있다. 남은 오차의 본질은 35개의 confident FN — **상폐 1년 전에도
재무비율상 정상으로 보이는 기업들**이다. 이것이 §9의 후속 stream이 부딪쳐야
할 진짜 천장이다.

---

## 8. 종합 (Synthesis): baseline이 실제로 말해주는 것

6개 진단을 종합하면:

1. **천장은 sharp하지 않고 넓다.** "0.2876"은 단일 seed의 최댓값이며 95%
   신뢰구간은 [0.18, 0.42], seed 평균은 0.270.
2. **9개 사전 실험은 통계적으로 실패하지 않았다.** 모두 baseline의 noise
   대역 안에 있다. 같은 평가 루프에서 스냅샷 모델 튜닝을 더 시도하는
   것은 추가 노력 대비 비효율적이다.
3. **데이터에 신호는 분명히 있다.** supervised RF가 모든 비지도 모델을
   bootstrap noise보다 훨씬 큰 차이로 능가한다. 재무비율 분포도 시간에
   걸쳐 안정적이다. 천장이 있다 해도 "데이터가 noise"라서는 아니다.
4. **평가는 불안정하다.** 두 가지 의미에서:
   (a) valid 연도(2023)가 우연히 2020~2023 중 가장 어려운 연도라
   hyperparameter 선택의 잘못된 anchor가 된다.
   (b) 2023→2024 shift가 2015–2022→2023 shift보다 크다. valid의 test로의
   전이가 부분적으로 깨져 있다.
5. **모델은 잘 보정되어 있고 합리적인 피처를 사용한다.** 유동성·수익성이
   지배하고 거시변수는 거의 noise. 부채비율만 의미 있는 분포 이동을 보임.
6. **남은 오차는 FP가 아니라 FN에 있다.** 35개 기업은 2024 재무비율이
   정상 기업과 구분 불가능했다. 어떤 미래 모델이든 깨야 할 진짜 경계선이다.

이 6가지가 종합적으로, **왜 프로젝트가 0.29에서 멈춰 있었는지**와 **어떤
종류의 개입이 실제로 의미 있게 움직일 수 있는지**를 동시에 설명한다.

---

## 9. 후속 Stream에 대한 시사점

로드맵에서 스케치된 4개 stream의 우선순위가 진단 결과에 비추어 재조정된다.

### 9.1 Stream 1 — 불규칙 시계열 모델링: **승격**

가장 강력한 증거는 **발견 6**이다. FN 기업들은 단일 스냅샷 재무비율이
"정상"으로 보인다. 이들을 진짜 정상 기업과 구분할 수 있는 방법은 오로지
**과거 3~5년의 재무비율 궤적(trajectory)**을 사용하는 것이다.

따라서 GRU-D, Neural CDE, mTAN, time2vec Transformer 같은 불규칙 시계열
모델이 가장 우선순위 높은 stream이 된다. 한국 기업의 공시 주기(연간 /
반기 / 분기 혼재)는 불규칙 시계열 방법론에 자연스럽게 맞아떨어진다.

### 9.2 Stream 2 — VAE 이상탐지: **위치 재조정 (피처 제공자로)**

§6의 순수 비지도 결과는 VAE 계열 monitoring statistic이 supervised RF를
PR-AUC에서 이기기 어려움을 보여준다. 그러나 같은 결과는 비지도 모델들이
신호의 일부는 독립적으로 추출할 수 있음(PR-AUC 0.11~0.16 vs 무작위 0.011)도
보여준다. 따라서 reconstruction error를 supervised 모델의 **추가 피처**로
사용하는 hybrid 방향은 여전히 유효하다.

베이지안 부트스트랩 기반 임계치 추정은 절대 PR-AUC와 무관하게 연구다운
확장이다 — UNIST DAL 시그니처와 직결.

### 9.3 Stream 3 — 생존 분석 / hazard 재프레이밍: **승격**

발견 4(a)는 fixed-N 라벨링의 취약점을 보여준다 — valid 연도 선택만으로
PR-AUC가 0.12 움직인다. Shumway(2001) 식의 이산시간 hazard 모형이나 Cox
time-varying 모형으로 재프레이밍하면 이 의존성이 사라진다 — 각 기업·연도가
자신의 censoring time으로 관측된다. 또한 "fixed_N1 vs N2" 비교
(`exp_018`) 자체가 baseline과 통계적으로 구분 불가하다는 사실은 fixed-N
설정 자체에 의문을 던지므로 hazard 모형이 본질적으로 더 깔끔하다.

### 9.4 Stream 4 — 분포 이동 적응: **현재 형태로는 우선순위 후순위**

§5에서 측정된 shift는 거시변수의 연도 key가 지배한다. KMM / CORAL / TENT
같은 표준 covariate-shift adaptation을 그 변수들에 적용해도 도움이 안 되는
이유는 단순하다 — 그것들은 §7에서 신호를 거의 운반하지 않는 변수들이다.

본 데이터에서 더 생산적인 "적응"은 **거시 피처를 재설계**하는 것이다 —
연도 key의 6개 절대값 대신 상대적·교호적 피처 (예: 부채비율 × VIX,
ROE × GDP)로 대체하여 모델이 절대 거시 상태가 아닌 기업·연도 간 contrast를
학습하도록 한다.

이는 일반 도메인 적응 알고리즘에 앞서 시도해야 하는 피처 엔지니어링 결정
이다.

### 9.5 요약 표

| Stream | S0 이전 우선순위 | S0 이후 우선순위 | 이유 |
|--------|------------------|-------------------|------|
| S1. 불규칙 시계열 | 권장 | **강력 권장** | FN 케이스는 trajectory 정보 필수 |
| S2. VAE 이상탐지 | 권장 | **권장 (피처 제공자로)** | 순수 비지도는 패배, reconstruction-as-feature는 유효 |
| S3. 생존 분석 | 선택 | **권장** | 취약한 fixed-N 의존성 제거 |
| S4. 분포 이동 적응 | 선택 | **후순위** | 거시 shift ≠ 신호 shift, 피처 재설계가 우선 |

---

## 10. 본 연구의 한계 (Threats to Validity)

- **Test 셋 크기.** test 2024 양성이 56개라 Bayesian bootstrap CI 자체가
  본질적으로 넓다. 진짜 개선이 있어도 이 셋만으로는 탐지가 불가능할 수
  있다. Walk-forward CV가 개발 단계에서 이를 일부 완화하지만, 더 강력한
  평가는 여러 test 연도 통합 또는 time-dependent AUC를 요구한다.
- **Hyperparameter 범위.** 본 연구에서 baseline RF의 hyperparameter는
  `exp_008` 값으로 모든 walk-forward fold에서 고정했다. 원칙적으로 fold별
  최적 hyperparameter가 달라질 수 있으며, fold별 절대 PR-AUC 평균이 달라질
  수 있다. 그러나 "exp_010~018이 모두 noise 안"이라는 결론 자체는 robust
  하다.
- **Bayesian 비교 vs paired DeLong.** exp_010~018의 예측 확률 벡터가
  저장되어 있지 않아 baseline posterior를 reference로 사용했다. 완전한
  paired DeLong 검정은 각 실험을 동일 seed로 재실행하여 매칭된 확률 벡터를
  얻어야 한다. 향후 특정 실험을 재방문할 때 가치 있는 확장이다.
- **ECOD 실패.** Python 3.13 + joblib + 비-ASCII home path 조합으로 ECOD
  학습이 실패했다. 나머지 4개 비지도 baseline이 일관된 결론을 주므로 §6의
  질적 결론에는 영향이 없다.

---

## 11. 재현성 (Reproducibility)

6개 진단을 다음과 같이 단일 명령으로 재현할 수 있다:

```powershell
.venv\Scripts\python.exe -m src.research.s0_diagnostic.run_s0_A1_bootstrap
.venv\Scripts\python.exe -m src.research.s0_diagnostic.run_s0_A2_walkforward
.venv\Scripts\python.exe -m src.research.s0_diagnostic.run_s0_A3_paired_tests
.venv\Scripts\python.exe -m src.research.s0_diagnostic.run_s0_A4_dist_shift
.venv\Scripts\python.exe -m src.research.s0_diagnostic.run_s0_A5_unsupervised
.venv\Scripts\python.exe -m src.research.s0_diagnostic.run_s0_A6_interpretation
.venv\Scripts\python.exe -m src.research.s0_diagnostic.aggregate
```

모든 스크립트는 기본 `seed=42`를 사용하며, 동일 seed에서 재실행 시 결과
JSON이 비트 단위로 일치한다. Windows 노트북 CPU 기준 전체 wall time은
약 25분이다 (A4, A6가 대부분 차지).

본 stream에서 추가로 설치한 의존성: `matplotlib`, `seaborn`, `pyod`, `shap`
(필요 시 `requirements_research.txt`로 별도 기록).

---

## 12. 참고 문헌

- Rubin, D.B. (1981). *The Bayesian bootstrap.* Annals of Statistics 9(1).
- Sun, X. & Xu, W. (2014). *Fast implementation of DeLong's algorithm.* IEEE Signal Processing Letters.
- Gretton, A. et al. (2012). *A kernel two-sample test.* JMLR 13.
- Liu, F.T., Ting, K.M., Zhou, Z.-H. (2008). *Isolation Forest.* ICDM.
- Breunig, M.M. et al. (2000). *LOF: Identifying density-based local outliers.* SIGMOD.
- Schölkopf, B. et al. (2001). *Estimating the support of a high-dimensional distribution.* Neural Computation.
- Goldstein, M., Dengel, A. (2012). *Histogram-based outlier score (HBOS).* KI.
- Li, Z. et al. (2022). *ECOD: Unsupervised outlier detection using empirical CDF.* IEEE TKDE.
- Guo, C. et al. (2017). *On calibration of modern neural networks.* ICML.
- Lundberg, S.M., Lee, S.-I. (2017). *A unified approach to interpreting model predictions.* NeurIPS.
- Shumway, T. (2001). *Forecasting bankruptcy more accurately: a simple hazard model.* Journal of Business.
