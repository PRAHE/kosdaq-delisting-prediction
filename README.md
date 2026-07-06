# KOSDAQ Delisting Early-Warning System
**DART 공시 재무데이터 기반 코스닥 상장폐지 조기경보 모델 (2015–2025)**

> "0.2876 PR-AUC 천장"이 noise band인지 진짜 한계인지를 묻는 것에서 출발해,
> 그 안에서 무엇이 실제로 작동했는지를 인과적으로 분해하는 연구.

---

## Overview

**연구 동기.**
코스닥 상폐는 소액주주 피해와 직결되지만, 사전 경보 체계는 여전히 단순 재무비율 스크리닝 수준에 머문다.
본 연구는 DART 공시 데이터를 활용해 상폐 1년 전을 조기에 탐지하는 신뢰성 중심 예측 모델을 구축한다.

**문제 정의.** 
코스닥 상폐 기업은 매년 소수(수십 개)이나, 그 사전 신호는
재무제표 어딘가에 있다. 본 연구는 DART API로 수집한 분기 재무제표(2015–2024,
약 1,500 종목)를 바탕으로 **상폐 1년 전**을 양성으로 정의하고 조기경보 모델을
학습·평가한다.

**데이터.** 
33개 재무비율(성장성/수익성/안정성/활동성) + 감사의견(DART) +
시장 OHLCV(FinanceDataReader) + 관리종목·주요사항보고서(KRX). 학습: 2015–2022
/ 검증: 2023 / 테스트: 2024. Test 양성 56건, 음성 5,255건(불균형 비율 약 94:1).

**핵심 결과.** 
기존 RF 스냅샷 baseline(PR-AUC 0.27) →
**Transformer + 감사의견(38 features) = PR-AUC 0.458**. Paired Bayesian
Bootstrap 기준 Δ +0.183, CI [+0.094, +0.279], P(best>base) = 1.000.

---

## Motivation

### 왜 PR-AUC / F2인가?

상폐는 희귀 사건이라 ROC-AUC는 다수 음성에 지배되어 모델의 실질적
판별력을 과장한다. PR-AUC는 양성 탐지 능력만을 직접 측정하며,
불균형 데이터에서 유일하게 신뢰할 수 있는 주지표다. F2는 Recall에
두 배 가중치를 줌으로써 "상폐를 놓치는 비용"이 "정상 기업을 잘못 경고하는
비용"보다 크다는 현실 목적함수를 반영한다.

### 왜 SMOTE를 기각했는가?

SMOTE는 학습 분포를 인위적으로 변형해 검증 PR-AUC를 부풀리지만,
시간 순서가 있는 재무 패널에서 미래 시점 피처가 합성 표본에 흡수되어
누수(data leakage)가 발생한다. 대신 Walk-forward CV로 시간 불변성을
보장하고, 멀티시드 앙상블과 Bayesian Bootstrap CI로 표본 불확실성을
명시적으로 정량화한다.

---

## Methodology

### 레이블 정의: fixed_N1 vs SR 비교

양성을 "상폐 정확히 N년 전 분기"로 정의하는 `fixed_N1` 방식을 default로
채택. 7개 대안 라벨링(rolling window, all-years 등)을 walk-forward +
bootstrap으로 비교한 결과 모두 PR-AUC ±3% 내 차이로 천장에 영향 없음을
확인, 라벨 재정의 방향을 기각했다. (`docs/research/research_sr_labeling_report.md`)

### Walk-forward CV 설계

시간 순서를 지키는 4-fold 확장 창(expanding window) 검증.
Fold 1: train 2015–2018 / test 2019, …, Fold 4: train 2015–2022 / test 2024.
단일 split에 비해 검정력을 4배로 높여 일관된 방향성을 측정한다.
(Fold별 양성 수: 12–56개 — 단일 점추정의 노이즈를 walk-forward 평균으로 안정화)

### 멀티시드 + Bayesian Bootstrap CI

단일 학습의 PR-AUC 95% CI 폭이 0.24에 달해(baseline [0.18, 0.42])
점추정 비교는 통계적으로 무의미하다. 5 seed 평균 확률을 Dirichlet(1,…,1)
사전으로 n=5,000 Paired Bayesian Bootstrap하여 Δ posterior를 직접 산출한다.

### Time2Vec + GRU / Transformer

불규칙 간격 분기 재무 패널에서 시간 임베딩(Time2Vec)을 positional encoding
대신 사용. GRU-D(결측 명시 모델링)는 부정적(과적합), Transformer(dim=64,
heads=4, 1-layer)가 walk-forward 4/4 fold에서 일관되게 우위. 기여 계층:
**감사 피처(+0.11) > 시계열 아키텍처(+0.06) > 알고리즘 선택(≈0)**.

---

## Results

### 주요 지표 (test 2024, 양성 56건)

| 모델 / 피처 구성 | PR-AUC | 95% CI | ROC-AUC | P@20 |
|---|---|---|---|---|
| RF baseline (33 fin.) | 0.270 | [0.180, 0.421] | 0.855 | — |
| RF + 감사의견 (38 feat.) | 0.367 | [0.260, 0.480] | 0.906 | 0.55 |
| Transformer (33 fin.) WF | 0.289 | — | — | — |
| Transformer + 감사 (38 feat.) | **0.458** | **[0.331, 0.586]** | **0.919** | **0.75** |
| RF + 시장 OHLCV (39 feat.) | 0.413 | — | 0.988* | — |

> *시장 피처의 ROC 0.988은 양성의 45%가 거래 동결(frozen) 종목인 artifact.
> 피처 제거 후 시계열 순효과 확인: Δ=+0.090, CI [+0.029, +0.158], P=0.998.

**Walk-forward (4-fold × 3-seed) 요약**

| 모델 | WF 평균 PR-AUC |
|---|---|
| RF baseline | 0.269 |
| Transformer (재무+감사, 38) | 0.408 |

상세 수치: `results/tables/best_ci_summary.json`, `results/tables/seed_results.csv`

---

## Limitations

**Valid-test PR-AUC 역전.** Walk-forward fold별로 valid PR-AUC가 test를
일관되게 상회한다. 원인: S0 진단에서 확인한 valid(2023)→test(2024) 분포
이동이 train→valid보다 커서 valid가 test의 예고편으로 기능하지 못한다.
조기경보 임계값 선택 시 valid 기준을 신뢰하면 test에서 성능이 하락할 수
있다.

**OoS 미성숙.** 2025년 test 실험에서 ROC-AUC는 0.94로 유지됐으나
PR-AUC가 0.06–0.10으로 붕괴했다. 세 모델이 동반 붕괴하고 ROC는 멀쩡한
패턴은 모델 실패가 아니라 2026년 상폐 라벨이 아직 기록되지 않은
censoring 현상이다. 진짜 OoS 검증은 2026년 말 라벨 성숙 후 재실행 예정.

**Confident FN.** 양성의 52%는 감사의견 적정 + 정상 거래 중인 채로
상폐된다. 감사 피처도 시장 피처도 이 사각지대를 포착하지 못한다.
재무 궤적(trajectory) 또는 hazard 모형의 영역으로 남겨둔다.

---

## How to Run

### 1. 환경 설정

```bash
pip install -r requirements.txt
```

`.env` 파일에 아래 키 설정:
```
DART_API_KEY=...
# DART OpenAPI 키 발급: https://opendart.fss.or.kr/
```

### 2. 데이터 수집

```bash
# DART 재무제표 수집 (raw JSON)
python -m src.data.dart_api --year 2024

# 감사의견 수집
python -m src.data.fetch_audit

# 분기 패널 데이터셋 빌드
python -m src.data.build_master_dataset
```

### 3. 학습 실행

```bash
# RF baseline
python -m src.train.run_rf_baseline

# Transformer walk-forward (best model)
python -m src.train.run_walkforward

# Bayesian Bootstrap CI (재현)
python -m src.evaluate.run_best_ci
```

### 4. 결과 확인

```bash
results/tables/best_ci_summary.json    # Δ posterior, P(best>base)
results/tables/seed_results.csv        # seed별 PR-AUC
results/figures/best_ci_comparison.png # posterior overlay 그림
```

---

## Project Structure

```
kosdaq-delisting-prediction/
├── src/
│   ├── data/          # DART API 수집, 전처리 파이프라인
│   ├── features/      # 재무비율, 감사/시장/관리종목 피처
│   ├── models/        # GRU, GRU-D, Transformer+Time2Vec
│   ├── train/         # 학습 루프, walk-forward CV
│   └── evaluate/      # PR-AUC, Bayesian Bootstrap CI
├── notebooks/         # EDA (기업별 분포, 결측 분석)
├── results/
│   ├── figures/       # PR-AUC posterior 비교, 학습 곡선
│   └── tables/        # best_ci_summary.json, seed_results.csv
├── docs/
│   └── research/      # 실험 단계별 보고서 (S0 → SR → S1 → A → 점검)
├── configs/           # 모델 하이퍼파라미터 (YAML)
└── requirements.txt
```

---

## Citation / Contact

이현지 · 광운대학교 컴퓨터정보공학부  
데이터: [DART OpenAPI](https://opendart.fss.or.kr/) (금융감독원), KRX 공시
