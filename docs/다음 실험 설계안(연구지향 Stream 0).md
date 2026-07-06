# 한국 상장폐지 예측 — 연구 지향 다음 실험 로드맵 (Stream 0 상세)

작성일: 2026-05-27
작성자: 이현지
배경: 광운대 → UNIST Data Analytics Lab (김성일 교수) 지원 포트폴리오 목적의 **연구 지향 실험 계획**

> 본 문서는 단순 metric 개선이 아닌 대학원 연구급 방법론(통계 엄밀성·이론 정합성·재현성)을 목표로 작성되었다.
> 우선 Stream 0 (진단 강화)을 상세 설계하고, Stream 1~4는 추후 결정용 스케치로 남겨둔다.

---

## Context (왜 지금 이 계획이 필요한가)

### 현재 한계
- exp_010~018 (Optuna RF/XGB/LGBM, FE, class_weight, SMOTE, Piotroski, fixed_N2) **9개 실험 모두 baseline RF PR-AUC 0.2876을 못 넘김**.
- 단일 seed · 단일 valid(2023, 양성 39개) · 단일 test(2024, 양성 56개) 단일 분할 → **PR-AUC 0.2876의 신뢰구간을 모름**.
- "baseline의 ceiling이 정말 0.29인가" 아니면 "평가 노이즈가 큰 것인가"가 미해결 → 다음 stream을 어디로 가야 할지 판단 불가.
- 분포 이동(2023→2024) 정량화·SHAP·calibration·비지도 이상탐지 baseline 모두 부재.

### 이 계획이 답하려는 것 (Research Questions)
1. **Baseline 0.2876은 통계적으로 어디까지가 ceiling인가?** (CI 부착)
2. **exp_010~018의 실패가 baseline과 통계적으로 유의한 차이였나?** 아니면 noise 안이었나?
3. **2023 → 2024 사이 어느 피처가, 어느 기업군에서 가장 분포가 이동했나?**
4. **비지도 이상탐지 baseline은 supervised RF를 능가하는가?** (= 문제의 본질이 anomaly detection인지 검증)
5. **baseline은 어디서 틀리는가?** (FP/FN의 SHAP, calibration)

→ 위 5개의 답이 이후 stream(시계열 / VAE / 생존분석 / 분포이동 적응) 중 어디에 투자해야 하는지를 결정한다. **진단 없는 후속 실험은 또 다른 exp_019가 될 위험.**

### UNIST DAL 연결 포인트
- 베이지안 부트스트랩 → S0의 CI 추정 방식 (랩 시그니처)
- 통계 지향 산업공학 랩 → 단발 metric이 아닌 **유의성·재현성**이 평가 포인트
- 비지도 이상탐지 baseline → 이후 VAE monitoring 계열 stream의 정당화 기반
- 불규칙 시계열(NCDE), VAE 기반 monitoring → Stream 1, 2에서 연결

### 일정 / 자원 제약
- 일정 미정, GPU는 교수님 요청 시 일부 가능
- 본 계획 **Stream 0은 CPU에서 전부 실행 가능** (트리 기반 + 통계검정 + IsolationForest 계열)
- 일정은 의도적으로 비워두고, 작업 단위(S0-A1 ~ S0-A6)별 산출물 완료 기준으로 진행 관리

---

## Current State Summary

데이터는 `combined_raw.csv` (55,681행 × 36피처: 27 재무비율 + 3 YoY + 6 거시경제), 라벨은 fixed_N1(상폐 정확히 1년 전, 124:1 불균형, train 283 / valid 39 / test 56 양성). 모델 코드는 `src/modeling/` 에 RF/XGB/LGBM/GBM 모듈화, `evaluate.py`가 PR-AUC + F1/F2/Recall@70/80% threshold 전략을 제공, GroupKFold(by stock_code)·seed 42 고정. 결과 저장은 `results/exp_*/`에 JSON.

**평가 인프라에서 부족한 것**: walk-forward CV, multi-seed, bootstrap CI, paired statistical test, 분포 이동 지표, calibration 지표, 비지도 이상탐지 baseline.

---

## Stream 0 — 진단 강화 & 평가 인프라 (상세)

### S0 산출물 (한눈에)

| ID | 산출 표/그림 | 답하는 RQ |
|---|---|---|
| S0-A1 | `bootstrap_ci_baseline.json` — PR-AUC 0.2876 ± 95% CI | RQ1 |
| S0-A2 | `walk_forward_results.csv` — 4 fold × 5 seed grid | RQ1 |
| S0-A3 | `paired_tests_vs_baseline.csv` — exp_010~018 Wilcoxon/DeLong p값 | RQ2 |
| S0-A4 | `distribution_shift.csv`, `psi_top_features.png` — PSI/KL/MMD | RQ3 |
| S0-A5 | `unsupervised_baselines.json` — IsoForest/LOF/OCSVM/ECOD vs RF | RQ4 |
| S0-A6 | `shap_summary.png`, `calibration_plot.png`, `fp_fn_cases.csv` | RQ5 |

각 산출은 baseline 1개에 대해서만 수행하면 충분 (RF, 33 피처, fixed_N1).

---

### S0-A1. Baseline의 신뢰구간 (Bootstrap CI)

**Research Question**: Baseline RF의 test PR-AUC 0.2876의 95% credible interval은?

**방법**:
- Test 셋(2024, 양성 56개)에 대해 **Bayesian bootstrap** (Rubin 1981) 1000회 resampling
  - 각 resample의 PR-AUC 계산 → posterior distribution
  - 95% credible interval(2.5%, 97.5%) 보고
- Frequentist bootstrap(percentile + BCa)도 병행 비교 (양성 56개라 분포 가정 영향 받음)
- 비교 baseline: 동일 seed (42)에서 학습된 현재 RF

**구현 위치**: `src/research/s0_diagnostic/bayesian_bootstrap.py`
- 함수: `bayesian_bootstrap_prauc(y_true, y_proba, n_boot=1000, seed=42) -> {mean, ci_lo, ci_hi, posterior_samples}`
- 재사용 가능하도록 `evaluate.py`의 결과 dict와 호환되는 입력 시그니처

**산출 파일**:
- `results/research_s0_diagnostic/A1_bootstrap_ci_baseline.json`
- `results/research_s0_diagnostic/A1_posterior_density.png`

**예상 결과 형태**:
```json
{"prauc_mean": 0.288, "ci95_lo": 0.235, "ci95_hi": 0.343, "n_boot": 1000, "method": "bayesian_dirichlet"}
```
→ 이 CI가 다른 모든 실험의 비교 기준선이 됨.

---

### S0-A2. Walk-Forward Cross-Validation

**Research Question**: 단일 valid(2023, 양성 39)의 평가 분산이 얼마나 큰가? 여러 시점으로 검증해도 baseline은 안정적인가?

**방법**:

| Fold | Train | Valid |
|---|---|---|
| 1 | 2015~2019 | 2020 |
| 2 | 2015~2020 | 2021 |
| 3 | 2015~2021 | 2022 |
| 4 | 2015~2022 | 2023 (현재 분할) |
| Test (단 한 번) | — | 2024 |

- 4 fold × 5 seed (42, 7, 13, 21, 100) = **20 측정**
- 각 측정에서 PR-AUC, F1, Recall, Precision, ROC-AUC 기록
- 보고: fold별 mean ± std, 전체 grand mean ± across-fold std

**구현 위치**: `src/research/s0_diagnostic/walk_forward_cv.py`
- 기존 `src/modeling/data_loader.py`의 train/valid 분할 로직 재사용 (연도 cutoff 인자만 추가)
- `evaluate.py`의 메트릭 함수 그대로 호출

**산출 파일**:
- `results/research_s0_diagnostic/A2_walk_forward_results.csv` (컬럼: fold, seed, prauc, f1, recall, ...)
- `results/research_s0_diagnostic/A2_fold_seed_grid.png` (heatmap)

**기대 효과**: 향후 다른 stream에서 hyperparameter 선택 시 단일 valid 대신 fold 평균 사용 → 선택 안정성↑.

---

### S0-A3. exp_010~018 vs Baseline 통계적 비교 (Paired Tests)

**Research Question**: exp_010~018의 PR-AUC 하락은 통계적으로 유의한 차이였나, 아니면 noise 범위 안이었나?

**방법**:
- 9개 실험 각각을 baseline과 동일한 4-fold × 5-seed 그리드로 **재실행** (단, 시간 부담을 줄이려면 동일 split의 prediction probability를 저장해놓고 sample-level paired test)
- 사용 통계검정:
  - **Wilcoxon signed-rank test** (paired, fold-level PR-AUC) — 분포 가정 약함, 작은 sample size에 적합
  - **DeLong test** (Sun & Xu 2014) — paired ROC-AUC 비교
  - **McNemar test** — threshold 후 confusion matrix 차이
- Multiple testing 보정: Holm-Bonferroni (9개 비교)

**구현 위치**: `src/research/s0_diagnostic/paired_tests.py`
- 함수: `paired_wilcoxon(prauc_a, prauc_b)`, `delong_test(y, proba_a, proba_b)`, `mcnemar(pred_a, pred_b, y)`

**산출 파일**:
- `results/research_s0_diagnostic/A3_paired_tests_vs_baseline.csv` (컬럼: exp_id, prauc_delta, wilcoxon_p, delong_p, mcnemar_p, holm_adjusted_p)
- 보고서 해석 1~2문단: 예: "exp_011~015는 baseline과 통계적으로 구분되지 않았다 (Holm p > 0.05). exp_014, exp_018만 유의하게 열등(p<0.01)."

**시간 부담 옵션**: 풀 재실행이 무겁다면, **fold-level PR-AUC를 4-fold × 1-seed = 4 값으로 줄여 Wilcoxon**만 수행해도 의미 있음.

---

### S0-A4. 분포 이동 (Distribution Shift) 정량화

**Research Question**: 2023 → 2024 사이 어느 피처가, 어느 기업군에서 가장 분포가 이동했나? exp_018 (valid 0.154 → test 0.01 붕괴)의 원인이 분포 이동이라는 가설을 지표로 입증할 수 있나?

**방법**:
- 피처별 **PSI (Population Stability Index)**: train vs valid, valid vs test, train vs test
  - PSI > 0.25 → significant shift (전통적 기준)
- 피처별 **KL divergence** (binned), **JS divergence** (대칭)
- 다변량 **Maximum Mean Discrepancy** (MMD, Gretton et al. 2012)
  - Gaussian kernel, bandwidth = median heuristic
  - permutation test로 p-value
- **섹터별, 규모별** breakdown — 어느 부분집합이 가장 이동했는가?

**구현 위치**: `src/research/s0_diagnostic/distribution_shift.py`
- 함수: `psi(p, q, n_bins=10)`, `kl_divergence(p, q)`, `js_divergence(p, q)`, `mmd_rbf(X, Y, bandwidth='median')`

**산출 파일**:
- `results/research_s0_diagnostic/A4_distribution_shift.csv` (피처별, 비교쌍별 지표)
- `results/research_s0_diagnostic/A4_psi_top_features.png` (top-10 shifted features bar plot)
- `results/research_s0_diagnostic/A4_mmd_per_sector.png`

**연결**: 이 결과는 향후 Stream 4 (분포 이동 적응)의 직접 입력. **현 단계에서는 진단만 하고, 적응 기법은 적용하지 않는다.**

---

### S0-A5. 비지도 이상탐지 Baseline

**Research Question**: "정상 기업만으로 학습한" 비지도 이상탐지가 supervised RF를 능가하는가? 그렇다면 본 문제의 본질은 anomaly detection.

**방법**:
- 학습: train 데이터 중 정상 기업(label=0)만 사용 → 비지도 모델 fit
- 평가: test (양성 56)에 대해 anomaly score → PR-AUC, ROC-AUC
- 비교 모델:
  1. **IsolationForest** (Liu et al. 2008)
  2. **LocalOutlierFactor** (LOF, Breunig et al. 2000)
  3. **OneClassSVM** (Schölkopf et al. 2001)
  4. **ECOD** (Li et al. 2022) — empirical CDF based, hyperparameter-free, recent
  5. **HBOS** (Goldstein & Dengel 2012) — histogram-based, fast
- 라이브러리: `pyod` (단일 인터페이스)
- 동일하게 multi-seed (5) × walk-forward (4 fold) 평가

**구현 위치**: `src/research/s0_diagnostic/unsupervised_baselines.py`

**산출 파일**:
- `results/research_s0_diagnostic/A5_unsupervised_baselines.csv` (모델별 PR-AUC mean±CI, vs RF Wilcoxon p)
- `results/research_s0_diagnostic/A5_anomaly_score_distribution.png`

**기대**: pyod 라이브러리로 구현 부담 작음. 결과가 RF와 근접하거나 능가하면 Stream 2 (VAE) 강력한 동기. RF가 크게 우세하면 supervised label이 본질적으로 더 정보 많다는 결론.

---

### S0-A6. 모델 해석 & Calibration

**Research Question**: Baseline은 어느 피처에 의존하고, 어디서 틀리는가? 확률 예측은 보정되어 있는가?

**방법**:
- **Permutation Importance** (sklearn, multi-seed로 CI 부착)
- **SHAP** (TreeExplainer for RF) — global summary plot + 상위 5 피처의 dependence plot
- **Calibration**:
  - Reliability diagram (10 bin)
  - Expected Calibration Error (ECE, Guo et al. 2017)
  - Brier score
- **FP / FN 사례 분석**: top-20 confident FP, top-20 confident FN의 SHAP 분해 + 정성 기록(섹터, 규모, 라벨년도)

**구현 위치**: `src/research/s0_diagnostic/interpretation.py`

**산출 파일**:
- `results/research_s0_diagnostic/A6_shap_summary.png`
- `results/research_s0_diagnostic/A6_permutation_importance.csv`
- `results/research_s0_diagnostic/A6_calibration_plot.png`
- `results/research_s0_diagnostic/A6_fp_fn_cases.csv`

---

### S0 통합 평가 프로토콜

모든 S0 산출은 다음 통합 표 한 줄로 보고된다:

| Model | Walk-fold | Seed | PR-AUC | PR-AUC 95% CI (Bayesian BS) | F1 | Recall | Precision | ECE | Wilcoxon p (vs baseline) |

`src/research/s0_diagnostic/aggregate.py`가 이 통합 표를 빌드한다.

---

### S0 코드 구조 (재사용 가능 자산)

```
src/research/s0_diagnostic/
├── walk_forward_cv.py         # 모든 stream이 공유할 splitter
├── bayesian_bootstrap.py      # CI 계산기
├── paired_tests.py            # Wilcoxon, DeLong, McNemar
├── distribution_shift.py      # PSI, KL, JS, MMD
├── unsupervised_baselines.py  # pyod 기반 4~5개 모델
├── interpretation.py          # SHAP, permutation, calibration
├── aggregate.py               # 모든 결과 통합 표
├── run_s0_A1_bootstrap.py
├── run_s0_A2_walkforward.py
├── run_s0_A3_paired_tests.py
├── run_s0_A4_dist_shift.py
├── run_s0_A5_unsupervised.py
└── run_s0_A6_interpretation.py

results/research_s0_diagnostic/
└── A1_*.* ~ A6_*.*  (위 표대로)
```

기존 `src/modeling/evaluate.py`, `src/modeling/data_loader.py`, `src/modeling/train_rf.py`는 **읽기 전용으로 import**하고 비파괴 확장.

---

### S0 검증 (Verification)

S0 완료 기준:
- [ ] A1 bootstrap CI 출력 — baseline PR-AUC 95% CI 보고
- [ ] A2 walk-forward 20개 측정 완료 — fold heatmap 생성
- [ ] A3 9개 exp_010~018 paired 비교 완료 — Holm 보정 p값 표
- [ ] A4 PSI/KL/MMD 표 + top-10 shifted features 그림
- [ ] A5 비지도 모델 5개 PR-AUC + vs RF Wilcoxon
- [ ] A6 SHAP, permutation, calibration, FP/FN 사례 50개

S0 보고서 (1차 산출, 다음 stream 결정의 입력):
- **제목 후보**: *"How tight is the 0.2876 ceiling? A diagnostic study of Korean delisting prediction"*
- 영문 abstract 1단락 + 한글 본문 3~4섹션 (Introduction, Setup, Diagnostic Findings, Implications for Next Steps)
- 보고서 위치: `docs/research_s0_diagnostic_report.md`

---

## Stream R — 라벨링·문제정의 점검 (S0 직후 추가)

### Context
- 추가 배경: 회의에서 "fixed_N1 라벨 1 데이터가 절대 부족(124:1)이라는 근본 문제. '상폐 예측'보다 '상폐 위험 검출'로 문제를 재정의하면 라벨 1을 확장할 여지가 있을지도 모름. 단 v3 사례(상폐 기업 모든 연도 = 1)는 valid 0.42 → test 0.234 붕괴를 보였으므로 **확장하되 왜곡 없이** 가능한 지점을 찾아야 함" 의견 제기.
- S0 결과의 발견 4(a) ("2023 valid는 비정상적으로 어렵고 fixed-N 라벨링은 valid 연도 선택에 취약")가 본 stream의 직접적 동기. S0 §9.3은 이미 Stream 3 (생존 분석)를 권장했으나, 그 전에 **단순 라벨링 변형 7가지**가 어디까지 가능한지를 먼저 점검하는 것이 효율적.

### Research Questions
- RQ-R1: train 라벨링 확장이 fixed_N1 ground truth에 대한 **test PR-AUC를 개선**하는가?
- RQ-R2: 어느 라벨링이 v3-style 붕괴(valid 과대, test 악화)를 **피하는가**?
- RQ-R3: 멀티-호라이즌·순서형·연속형 라벨링은 binary fixed_N1 대비 정보를 더 잘 사용하는가?
- RQ-R4: 라벨링 변경이 walk-forward fold 안정성에 어떤 영향을 주는가?
- RQ-R5: "위험 검출" 관점의 운영 지표(top-K precision, lift)에서 어느 라벨링이 우수한가?

### Methodology
- **Train 라벨링만 변경, valid/test는 fixed_N1로 고정** (공정 비교 핵심, 사용자 confirm)
- 모든 라벨링에 동일한 baseline RF 적용 (모델 변동성 제거)
- S0 인프라 재사용: walk-forward 4 fold + 5 seed + Bayesian bootstrap CI
- 평가 지표: PR-AUC ± CI, ROC-AUC, top-20/50 precision, top-K lift, walk-forward stability(std)

### 7개 실험 매트릭스

| ID | Train 라벨링 | 양성 비율 (예상) | 의미 |
|----|--------------|------------------|------|
| R-L1 | fixed_N1 (현재 baseline) | 1.0x (283) | 상폐 정확히 1년 전만 |
| R-L2 | rolling_H12 (≤1년 내) | 1.0~1.5x | 1년 내 상폐 발생 |
| R-L3 | rolling_H24 (≤2년 내) | 2x | 2년 내 상폐 발생 |
| R-L4 | fixed_N1 ∪ N2 ∪ N3 | ~3x | 상폐 1·2·3년 전 모두 1 |
| R-L5 | ordinal 4-class (0=정상,1=N3,2=N2,3=N1) | full | 순서형 위험도 |
| R-L6 | continuous risk score exp(-Δt/τ) | full | 위험도 회귀 |
| R-L7 | v3-style 상폐 기업 모든 연도 = 1 | ~7x | negative anchor (이미 실패) |

### 코드 구조
```
src/research/sr_labeling/
├── labelings.py          # 7개 라벨링 변환 함수
├── train_eval.py         # train_labeling 학습, fixed_N1 평가
├── run_sr_all.py         # 전체 grid 실행
└── aggregate.py          # 결과 통합

results/research_sr_labeling/
└── R_*.* (grid CSV, comparison PNG, summary JSON)
```

기존 `src/research/s0_diagnostic/{baseline.py, walk_forward_cv.py, bayesian_bootstrap.py}`를 import.

### Stream R 검증
- [ ] 7개 라벨링 × 4 fold × 5 seed grid 실행
- [ ] 각 라벨링의 fixed_N1 ground truth에 대한 test PR-AUC ± CI
- [ ] walk-forward fold-level PR-AUC variance
- [ ] top-20/50 precision (위험 검출 운영 지표)
- [ ] v3 붕괴 패턴 (valid-test 격차) 비교

### 산출물
- `results/research_sr_labeling/R_results_grid.csv`
- `results/research_sr_labeling/R_pr_auc_comparison.png`
- `results/research_sr_labeling/R_top_k_metrics.csv`
- `results/research_sr_labeling/R_summary.json`
- `docs/research_sr_labeling_report.md` — 진단 보고서 + 최종 라벨링 권장

### 다음 stream에의 영향
- R-L4 (fixed_N1∪N2∪N3) 또는 R-L5 (ordinal)가 우수 → S3 (생존 분석)로 자연 연결
- R-L2/L3 (rolling H) 우수 → S1 (시계열 모델링)에 multi-horizon 입력으로 활용
- 모든 변형이 baseline 미만 → "라벨링은 문제가 아니다" 결론 → S1/S3 직진
- R-L7이 valid-test 격차를 드러내면 v3 실패 메커니즘 재현 + 정량화

---

## Stream 1~4 (추후 결정용 스케치)

> S0 완료 후 결과에 따라 우선순위 재조정. 현 단계에서는 **방향만** 표시.

| Stream | 1줄 요약 | UNIST DAL 키워드 | S0 어떤 결과가 정당화하는가 |
|---|---|---|---|
| **S1. 불규칙 시계열 분류** | GRU-D / Neural CDE로 분기·반기 혼재 그대로 모델링 | NCDE, irregular TS | S0가 "snapshot 모델은 ceiling"을 입증하면 본 stream 필수 |
| **S2. VAE 이상탐지** | 정상 패널로 VAE fit → Hotelling T²/SPE/NLL 이상점수 + 베이지안 BS threshold | VAE monitoring, Bayesian BS | S0-A5에서 비지도 모델이 supervised에 근접/능가하면 강력 정당화 |
| **S3. 생존 분석** | Cox time-varying / Discrete hazard / DeepSurv로 fixed-N 의존성 제거 | panel time series, censored data | S0가 "라벨 정의에 결과가 민감"을 보이면 본 stream |
| **S4. 분포 이동 적응** | KMM/CORAL/TENT/Temperature scaling | distribution shift, calibration | S0-A4에서 PSI/MMD가 큰 shift를 정량화하면 본 stream |

GPU 의존도:
- S1: GRU-D는 CPU 가능, Neural CDE는 GPU 필요
- S2: vanilla VAE CPU 가능, LSTM-VAE GPU 권장
- S3: 통계 모델 CPU, DeepSurv GPU 권장
- S4: 대부분 CPU 가능

---

## 통합 검증 (전체 Verification)

1. **로컬 검증**: S0 모든 스크립트를 단일 명령으로 재실행
   ```powershell
   python -m src.research.s0_diagnostic.run_s0_A1_bootstrap
   python -m src.research.s0_diagnostic.run_s0_A2_walkforward
   python -m src.research.s0_diagnostic.run_s0_A3_paired_tests
   python -m src.research.s0_diagnostic.run_s0_A4_dist_shift
   python -m src.research.s0_diagnostic.run_s0_A5_unsupervised
   python -m src.research.s0_diagnostic.run_s0_A6_interpretation
   python -m src.research.s0_diagnostic.aggregate
   ```
2. **재현성 점검**: 모든 스크립트는 seed list를 인자로 받고, 동일 seed로 재실행 시 출력 JSON이 byte 단위 동일해야 함.
3. **결과 해석 점검**: S0 완료 후 보고서 초안(`docs/research_s0_diagnostic_report.md`) 작성 → 본 plan의 5개 RQ에 모두 답이 있는지 self-review.
4. **다음 stream 결정 회의**: S0-A4(분포이동)/A5(비지도) 결과를 보고 S1/S2/S3/S4 중 어느 stream에 먼저 투자할지 정한다.

---

## Critical Files (구현 시 참조)

- 평가 기반: `src/modeling/evaluate.py` (메트릭 함수 import)
- 분할 기반: `src/modeling/data_loader.py` (연도 cutoff 확장)
- baseline 학습: `src/modeling/train_rf.py` (RF 하이퍼파라미터 그대로 import)
- 결과 저장 패턴: `src/modeling/run_all.py` (JSON 스키마 호환)
- 원천 데이터: `preprocess/data/processed/combined_raw.csv`
- 기존 실험 결과: `results/exp_010_*` ~ `results/exp_018_*` (S0-A3 paired test 입력)

추가 의존성 (requirements_research.txt):
- `pyod` (S0-A5 비지도)
- `shap` (S0-A6)
- `scipy.stats` (Wilcoxon, McNemar — 이미 있음)
- `lifelines` (S3 시점에 추가)
- 신규 딥러닝 의존성은 Stream 0에서는 불필요
