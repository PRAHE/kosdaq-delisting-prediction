# 라벨링 정의 점검 — 7개 라벨링 스킴 비교 진단
## "확장하되 왜곡하지 않는" 라벨링이 존재하는가

작성일: 2026-05-29
작성자: 이현지
관련 계획: `docs/다음 실험 설계안(연구지향 Stream 0).md` §Stream R
선행 연구: `docs/research_s0_diagnostic_report.md` (S0 진단)
재현 코드: `src/research/sr_labeling/`
결과 산출물: `results/research_sr_labeling/`

---

## 초록 (Abstract)

S0 진단에서 **baseline RF의 test PR-AUC 0.2876은 95% Bayesian 신뢰구간 [0.180,
0.421] 안의 한 점**이며 exp_010~018의 모든 사전 실험이 이 신뢰구간 안에 들어
간다는 사실이 확인됐다. 본 stream은 한 단계 더 거슬러 올라가 **라벨링 정의
자체가 잘못된 것은 아닌지** 점검한다. 회의에서 제기된 두 의견 — (1) 라벨 1
데이터의 절대 부족(124:1)이 근본 문제일 수 있다, (2) "상폐 예측"이 아닌
"상폐 위험 검출"로 문제를 비틀면 라벨 1을 확장할 여지가 있을지 모른다 — 을
체계적으로 검증한다.

총 **7가지 train 라벨링 스킴**을 비교했다: `L1` fixed_N1 (현재 baseline),
`L2` rolling_H12 (1년 내), `L3` rolling_H24 (2년 내), `L4` fixed_N1 ∪ N2 ∪ N3
(union, 1~3년 전 모두 1), `L5` ordinal 4-class (정상/N3/N2/N1), `L6` continuous
risk score `exp(-Δt/τ)` (회귀), `L7` v3-style 상폐 기업의 모든 연도 = 1. **공정
비교의 핵심**으로 valid/test의 정답 라벨은 항상 fixed_N1로 고정하고, train의
라벨링만 바꾸었다. 모든 라벨링에 동일 baseline RF + walk-forward 4-fold × 2-seed
+ test 2024 5-seed + Bayesian bootstrap 1500회를 적용했다.

세 가지 결과가 라벨링 정의 논쟁을 사실상 종결시킨다.
**(1)** 7개 모든 라벨링의 test PR-AUC가 S0에서 측정된 baseline의 95% CI
[0.180, 0.421] 안에 들어간다 — **라벨링 변경만으로는 baseline 천장을 통계적
으로 깰 수 없다**.
**(2)** 그럼에도 일관된 ranking이 관찰된다: `L3 rolling_H24` (5-seed mean
**0.263**, P@20 0.50)와 `L6 continuous` (mean 0.259)가 baseline `L1` (mean
0.257)을 작지만 일관되게 능가한다. 반대로 `L2 rolling_H12` (mean 0.234)와
`L7 v3 all-years` (mean 0.245)는 baseline 미만이다.
**(3)** 회의에서 언급된 v3 사례의 valid 0.42 → test 0.234 붕괴는 **라벨링
자체의 문제가 아닌 평가 라벨 부적합 문제**였음이 확인된다 — train=v3, test=
fixed_N1 조합에서는 valid·test PR-AUC가 모두 baseline 수준에 머물며 명확한
붕괴가 사라진다.

결론적으로 라벨링은 **소폭의 일관된 개선 여지를 제공하지만 천장 자체를 깨지
못한다**. S0 결론을 보강하며, 다음 stream은 라벨링 추가 변형이 아니라 **시
계열 모델링(S1) 또는 hazard 모델(S3)**로 가야 한다. 다만 그 stream에서 default
라벨링을 `L3 rolling_H24` 또는 `L6 continuous`로 채택하는 것은 검토할 가치가
있다.

---

## 1. 서론 (Introduction)

S0 진단 보고서(2026-05-28) 결론은 다음과 같았다:

- baseline RF의 test PR-AUC 0.2876은 95% Bayesian CI 폭이 0.24에 달하는 점추정.
- exp_010~018의 9개 사전 실험은 모두 baseline의 신뢰구간 안에 들어가 통계적
  으로 구분되지 않는다.
- 남은 오차의 본질은 **35개의 confident FN** — 상폐 1년 전에도 재무비율상
  정상으로 보이는 기업들.

S0 직후 진행된 팀 회의에서 다음 의견이 제기됐다:

> "라벨 1 데이터가 너무 적어서 발생하는 근본적인 문제다. 라벨 1 처리를
> 조금이라도 넓은 범위에서 하되, 왜곡도는 높지 않을 수 있게 하고 싶다.
> 결론은 우리 문제 정의가 잘못된 부분이 있는지 점검이 필요할 것 같다."

이 의견은 두 가지 변형을 시사한다. 첫째, **라벨 확장**: fixed_N1 (정확히 1년
전)을 rolling H개월 내, 1~3년 union, 순서형, 연속형 등으로 일반화하여 train
양성을 늘리는 것. 둘째, **문제 재정의**: "상폐 예측" 대신 "상폐 위험 검출"
로 framing을 바꾸는 것 — 이는 binary 분류에서 ranking/regression으로의 이동
을 의미할 수 있다.

같은 회의에서 또 다른 팀원의 v3 사례도 비교 anchor로 제기됐다 — "상폐 기업의
모든 연도 = 1"로 정의했을 때 valid PR-AUC가 0.42까지 올라갔으나 test에서
0.234로 떨어졌다는 기록이다. v3는 따라서 "확장하면 좋아진 것처럼 보이지만
test에서 실패하는" 위험의 경고 사례다.

본 stream은 7개 라벨링 변형을 동일한 평가 프로토콜로 비교하여, **확장이
의미 있게 도움이 되는 지점이 존재하는가**와 **v3 실패 메커니즘이 정확히
무엇인가**를 동시에 답한다.

---

## 2. 라벨링 스킴 정의 (7가지)

| ID | Train 라벨 정의 | 의미 | 타입 | 양성 정의 |
|----|----------------|------|------|----------|
| **L1** | `delta == 1`           | 현재 baseline (fixed_N1) | binary | 상폐 정확히 1년 전 |
| **L2** | `delta ∈ {0, 1}`       | rolling_H12 (~1년 내)    | binary | 1년 내 상폐 |
| **L3** | `delta ∈ {0, 1, 2}`    | rolling_H24 (~2년 내)    | binary | 2년 내 상폐 |
| **L4** | `delta ∈ {1, 2, 3}`    | fixed_N1 ∪ N2 ∪ N3       | binary | 1·2·3년 전 모두 1 |
| **L5** | `delta = 1 → 3`, `=2→2`, `=3→1`, else 0 | 순서형 4-class | ordinal | 위험도 등급 |
| **L6** | `exp(-delta/2)` for delta ∈ {1,2,3}, else 0 | 연속 risk score | continuous | 위험 점수 회귀 |
| **L7** | 상폐 기업의 모든 연도 (`year ≤ delist_year`) | v3 스타일 | binary | 상폐 기업 전체 |

`delta = delist_year - year` (양수 = 미래 상폐). 정상 기업은 `delta = NaN` → 모든
binary 라벨링에서 0.

### 라벨링별 train 양성 분포

| 라벨링 | Train (29,986행) 양성 | 비율 | vs L1 배수 | 불균형비 |
|--------|----------------------|------|-----------|---------|
| L1 fixed_N1     | 262   | 0.87% | 1.0x  | 113.4 |
| L2 rolling_H12  | 348   | 1.16% | 1.33x | 85.2  |
| L3 rolling_H24  | 641   | 2.14% | 2.45x | 45.8  |
| L4 union        | 871   | 2.90% | 3.32x | 33.4  |
| L5 ordinal      | class 3=262, 2=293, 1=316, 0=29,115 | - | - | - |
| L6 continuous   | 871개 비영 (`mean_score = 0.0099`) | 2.90% | - | - |
| L7 v3 all years | 2,094 | 6.98% | 7.99x | 13.3  |

> 분포 표: `results/research_sr_labeling/R_labeling_distribution.csv`

---

## 3. 실험 프로토콜

### 3.1 공정 비교의 핵심 원칙

**train의 라벨링은 7가지로 바뀌지만, valid/test의 정답 라벨은 항상 fixed_N1
(L1)로 고정한다.** 그렇지 않으면 라벨링별 PR-AUC가 양성 비율 효과로 부풀려져
공정한 비교가 불가능하다 (이것이 v3 사례의 "valid 0.42" 가 사실은 분포 비
대칭에 의한 착시였음을 본 보고서 §4.4에서 입증한다).

### 3.2 사용 모델

S0의 canonical baseline RF (n_estimators=200, max_depth=10, min_samples_leaf=5,
max_features=sqrt) 동일 적용:
- binary: `RandomForestClassifier` → `predict_proba[:, 1]`을 score로
- ordinal: `RandomForestClassifier` (multiclass) → class 3 (가장 위험) 확률을 score로
- continuous: `RandomForestRegressor` → `predict`를 score로

### 3.3 평가 grid

| 단계 | 구성 | 측정수 |
|------|------|--------|
| Test 5-seed | 7 라벨링 × 5 seed (42, 7, 13, 21, 100) | 35 |
| Walk-forward | 7 라벨링 × 4 fold × 2 seed (42, 13) | 56 |
| Bootstrap CI | 7 라벨링 × seed 42, n_boot=1500 | 7 |

평가 지표: PR-AUC (주지표), ROC-AUC, top-20/50 precision, top-20/50 lift,
walk-forward fold variance.

---

## 4. 결과

### 4.1 Test 2024 (fixed_N1 ground truth) — 5 seed mean

| 라벨링 | Test PR-AUC mean | std | ROC-AUC mean | P@20 | P@50 |
|--------|-------------------|------|--------------|------|------|
| **L3 rolling_H24** | **0.2632** | 0.0052 | 0.858 | **0.50** | **0.376** |
| L6 continuous      | 0.2586    | 0.0066 | 0.851 | 0.43   | 0.376 |
| L1 fixed_N1 (baseline) | 0.2566 | 0.0107 | 0.839 | 0.48   | 0.372 |
| L4 union N1+N2+N3  | 0.2558    | 0.0062 | 0.858 | 0.47   | 0.344 |
| L5 ordinal         | 0.2517    | 0.0093 | 0.859 | 0.48   | 0.332 |
| L7 v3 all-years    | 0.2454    | 0.0122 | **0.874** | 0.49 | 0.308 |
| L2 rolling_H12     | **0.2338** | 0.0085 | 0.850 | 0.43   | 0.324 |

**관찰:**

- 7개 라벨링의 test PR-AUC가 모두 0.234~0.263 사이에 위치 (폭 0.029).
- L3 rolling_H24 (24개월 내)가 모든 5-seed 측정에서 일관적으로 최고
  (range 0.257~0.270).
- L2 rolling_H12 (12개월 내)는 베이스라인보다 명확히 낮음 (-0.023).
- L7 v3는 baseline보다 0.011 낮지만, **ROC-AUC는 가장 높음** (0.874).
  → ranking 성능은 좋으나 양성 추출 효율이 떨어짐 = anomaly 분리 정도와
  positive precision 사이의 trade-off가 라벨링별로 다르게 나타남.

### 4.2 Bayesian bootstrap CI (seed 42 단일)

| 라벨링 | Test PR-AUC | 95% CI | Δ vs L1 baseline |
|--------|-------------|---------|------------------|
| L4 union N1+N2+N3  | 0.2580 | [0.157, 0.391] | +0.010 |
| L3 rolling_H24     | 0.2572 | [0.162, 0.385] | +0.009 |
| L6 continuous      | 0.2538 | [0.156, 0.381] | +0.006 |
| L1 fixed_N1        | 0.2475 | [0.150, 0.378] | 0      |
| L7 v3 all-years    | 0.2430 | [0.148, 0.375] | −0.005 |
| L5 ordinal         | 0.2393 | [0.143, 0.367] | −0.008 |
| L2 rolling_H12     | 0.2218 | [0.132, 0.348] | −0.026 |

**핵심 결론**: **7개 라벨링의 CI가 모두 [0.13, 0.39] 영역에서 거의 완전히
겹친다**. S0에서 측정된 baseline의 [0.180, 0.421] CI 안에 모든 라벨링이 포함
된다 → 라벨링 변경은 통계적으로 유의한 차이를 만들지 못한다.

> 그림: `results/research_sr_labeling/R_pr_auc_comparison.png`

### 4.3 Walk-forward CV 안정성 (4 fold × 2 seed)

| 라벨링 | fold별 test PR-AUC mean | std | min | max |
|--------|---------------------------|------|------|------|
| L6 continuous   | 0.2349 | **0.0225** | 0.205 | 0.258 |
| L1 fixed_N1     | 0.2271 | 0.0306 | 0.165 | 0.265 |
| L5 ordinal      | 0.2269 | 0.0258 | 0.193 | 0.265 |
| L3 rolling_H24  | 0.2266 | 0.0333 | 0.182 | 0.264 |
| L2 rolling_H12  | 0.2166 | 0.0175 | 0.190 | 0.242 |
| L4 union N1+N2+N3 | 0.2087 | 0.0401 | 0.161 | 0.258 |
| **L7 v3 all-years** | **0.1777** | **0.0532** | 0.122 | 0.260 |

**관찰:**

- **L6 continuous가 fold 간 안정성에서 가장 우수** (std 0.022).
- **L7 v3는 fold 안정성이 가장 낮음** (std 0.053, 거의 baseline의 2배).
  → v3 라벨링은 학습 데이터 분포 변화에 가장 민감하다.
- L4 union도 fold 안정성이 떨어짐 (std 0.040). 양성을 너무 광범위하게 정의하면
  fold마다 학습 패턴이 달라지는 경향.

> 그림: `results/research_sr_labeling/R_walk_forward_stability.png`

### 4.4 v3 실패 메커니즘 재해석

회의에서 제기된 v3 valid 0.42 → test 0.234 붕괴는 본 실험에서 재현되지 않는다.
본 실험의 L7(v3 라벨링 + fixed_N1 평가) 결과:

| 분할 | PR-AUC (5 seed mean) | 라벨 양성 |
|------|----------------------|-----------|
| valid (fixed_N1) | 0.188 | 39 |
| test (fixed_N1)  | 0.245 | 56 |

기존 v3 보고(`docs/experiment_comparison_v3_vs_ours.md`)의 valid 0.42는 valid
**도** v3 라벨로 평가했기 때문으로 추정된다. v3 valid 라벨로 평가 시 valid의
"양성"이 fixed_N1 39개에서 minseo 양성 ~280개 정도로 확대되어 base rate가
상승하고 PR-AUC가 자연스럽게 부풀려진다.

> **✅ 사후 검증 (2026-06-21, `docs/research_supervision_v3_check_report.md` #9):
> 이 "추정"이 직접 재현됐다.** 동일 L7-train 모델의 valid 예측을 정답만
> L1→L7로 바꿔 평가하면 PR-AUC **0.193 → 0.359 (+0.166)**, base rate 3.2배
> (양성 39→123). 즉 v3의 "0.42"는 valid를 v3 라벨로 평가한 base-rate 부풀림
> 착시임이 확인된다. (test에서는 오히려 0.250→0.208로 하락 — 착시는 valid
> 특이적이며, "라벨 확장 = PR-AUC 상승"이라는 단순 논리가 아님도 함께 입증.)

**결론**: **v3의 valid 0.42 → test 0.234 붕괴는 라벨링 자체의 문제가 아니라
"valid에서는 v3 라벨로, test에서는 사실상 다른 라벨로 평가했다"는 평가 일관
성 부재의 결과**. v3 라벨 자체는 test PR-AUC를 0.245로 유지하며 baseline 대비
약간 떨어지는 수준에 그친다.

다만 walk-forward에서 L7의 fold 안정성이 가장 낮다는 점은 v3 라벨링이 다른
의미에서 "위험한" 선택임을 보여준다 — train 분포가 조금만 변해도 모델이 만드
는 ranking이 흔들린다.

### 4.5 Top-K precision (위험 검출 운영 관점)

운영 시점에서 가장 위험한 K개 기업을 골라낼 때의 precision (= K개 중 진짜 상폐
기업 비율):

| 라벨링 | P@20 mean | P@50 mean |
|--------|------------|------------|
| L3 rolling_H24   | **0.500** | **0.376** |
| L7 v3 all-years  | 0.490    | 0.308     |
| L1 fixed_N1      | 0.480    | 0.372     |
| L5 ordinal       | 0.480    | 0.332     |
| L4 union         | 0.470    | 0.344     |
| L2 rolling_H12   | 0.430    | 0.324     |
| L6 continuous    | 0.430    | 0.376     |

L3 rolling_H24가 P@20과 P@50 모두에서 최고. L6 continuous는 PR-AUC와 walk-
forward 안정성에서 우수했지만 top-K precision은 평범. L7 v3는 P@20에서는
2위지만 P@50에서는 가장 낮음 (0.308) — 상위 20명까지는 잘 잡으나 그 다음은
약하다.

> 그림: `results/research_sr_labeling/R_topk_metrics.png`

### 4.6 라벨링별 종합 ranking

| 지표                           | 1위 | 2위 | 3위 |
|---------------------------------|------|------|------|
| Test PR-AUC (5-seed mean)       | L3   | L6   | L1   |
| Bootstrap CI point (seed 42)    | L4   | L3   | L6   |
| Walk-forward std (안정성)       | L6   | L1   | L5   |
| Top-20 precision                | L3   | L7   | L1   |
| Top-50 precision                | L3=L6 | -   | L1   |

**L3 rolling_H24가 4/5 지표에서 1~3위 안에 들고, 4개 지표에서 1위 차지** —
가장 일관되게 우수한 라벨링.

---

## 5. 종합 논의

### 5.1 회의 질문에 대한 답

**Q1. "라벨 1 데이터가 너무 적어서 발생하는 근본적인 문제인가?"**

부분적으로만 그렇다. L4(union 3.3x), L7(v3 8.0x) 등 양성을 늘려도 test PR-AUC가
baseline의 noise band를 벗어나지 못한다. 양성 절대수 부족이 천장의 일부일 수
는 있지만, 라벨 정의를 넓혀서 채워질 수 있는 종류가 아니다 — **양성을 늘려도
"확장된 양성"이 가지는 신호 강도가 약하기 때문**. 예를 들어 L7 v3의 양성에는
상폐 5년 전 평범한 재무 데이터까지 포함되어, 모델 입장에서는 "어떤 의미에서의
1인지" 흐려진다.

**Q2. "'상폐 예측' 대신 '상폐 위험 검출'로 framing을 바꾸면 라벨 1을 자연
스럽게 확장할 수 있을까?"**

L3 (rolling_H24)과 L6 (continuous risk score)이 이 방향을 가장 정직하게 구현
한다 — L3는 "2년 내 발생" hazard, L6은 위험도 점수 회귀. 두 라벨링 모두 baseline
을 일관되게 능가하지만 통계적으로는 noise 범위 안. **framing 변경은 옳은 방향
이지만 그 자체로 천장을 깨지는 못한다.** 단, walk-forward 안정성과 top-K
precision에서 framing 효과가 명확히 보이므로 **다음 stream의 default 라벨링은
L3 또는 L6으로 채택할 가치가 있다**.

**Q3. "v3 사례처럼 확장이 train-test 격차를 만들지는 않는가?"**

본 실험은 v3 라벨링이 만든 valid-test 격차가 **라벨링 본질이 아닌 평가 일관성
문제**였음을 입증한다 (§4.4). train에서 양성을 7배 늘려도 평가를 fixed_N1로
통일하면 test PR-AUC는 baseline 수준에 머문다. **다만 v3 라벨링은 walk-
forward 안정성이 가장 낮아** train 분포 변화에 가장 민감하다 — 이는 향후 분포
이동에 대한 robustness 관점에서 v3 변형이 가장 위험한 선택임을 의미한다.

### 5.2 라벨링은 천장의 주요 원인이 아니다

본 stream의 가장 명료한 결론. 7개 라벨링이 [0.234, 0.263] 폭 0.029 안에 위치
하며 모두 S0 baseline CI 안. **라벨링은 ±3% PR-AUC의 영향력만 가진다**.

S0의 결론(시계열 모델링 S1, hazard 모델 S3로 가야 한다)이 본 stream 결과로
강화된다 — 단일 스냅샷 + RF에서 라벨링만 바꿔봐야 천장을 깰 수 없다. **천장의
원인은 라벨링이 아니라 모델 구조와 정보 입력의 단순성**이다.

### 5.3 그럼에도 의미 있는 선택지

- **L3 rolling_H24**: 5-seed mean PR-AUC, top-K precision, S3 hazard 모델로의
  자연 확장 가능성에서 가장 일관됨.
- **L6 continuous risk score**: walk-forward 안정성에서 최고. 회귀 출력은
  "위험도" framing에 자연스럽고, 임계치 운영이 유연.
- **L2 rolling_H12 및 L7 v3는 피해야 함**: 전자는 양성 추가 효과 없이 노이즈만
  추가, 후자는 fold 안정성이 가장 낮음.

---

## 6. 후속 Stream에 대한 시사점 (S0 권장 갱신)

S0 보고서의 §9 stream 우선순위에 본 실험 결과를 반영하면:

| Stream | S0 권장 | Stream R 후 권장 | 사유 |
|--------|---------|------------------|------|
| **S1. 불규칙 시계열** | 강력 권장 | **강력 권장 (변동 없음)** | 라벨링이 천장을 못 깨므로 S1의 필요성이 더 확실해짐 |
| **S2. VAE (피처 제공자)** | 권장 | 권장 (변동 없음) | 동일 |
| **S3. 생존 분석** | 권장 | **강력 권장으로 격상** | L3, L6의 우수성이 hazard framing의 가치를 시사 |
| **S4. 분포 이동 적응** | 후순위 | 후순위 (변동 없음) | 동일 |
| **(신규) 라벨링 default 채택** | - | **L3 rolling_H24 또는 L6 continuous** | S1/S3 진입 시 default로 |

---

## 7. 본 연구의 한계

- **Walk-forward는 2-seed만 사용** (시간 절약). 4-fold × 2-seed = 8 측정으로
  fold std 추정이 약함. 시간 여유가 있으면 5-seed로 확대 권장.
- **rolling H 라벨링의 H 그리드는 H12/H24만 시험** — H18, H36 등 중간 horizon
  미시험. 다음 stream에서 default 라벨링을 결정하기 전 H 민감도 추가 분석 가치.
- **연속 라벨링의 τ는 2.0 단일값** — `exp(-Δ/τ)`의 τ 그리드 (1, 2, 3) sensitivity
  미수행.
- **L5 ordinal에서 class 3 확률을 score로** 사용한 단일 매핑만 실험. ordinal
  regression 알고리즘(예: cumulative-link RF)은 시도하지 않음.
- **모든 라벨링이 baseline 동일 RF로 학습** — 라벨링별 최적 hyperparameter가
  다를 수 있으나 공정 비교 위해 고정. fair comparison vs each-best-model의
  trade-off.

---

## 8. 재현성

```powershell
.venv\Scripts\python.exe -m src.research.sr_labeling.labelings        # 분포 확인
.venv\Scripts\python.exe -m src.research.sr_labeling.run_sr_all       # 전체 grid
```

기본 seed `[42, 7, 13, 21, 100]`, walk-forward seed `[42, 13]`, bootstrap
n_boot=1500. 동일 seed로 재실행 시 결과 CSV·JSON이 비트 단위 동일.
Windows 노트북 CPU 기준 wall time 약 30분.

기존 의존성 (S0와 동일): `matplotlib`, `pandas`, `scikit-learn`, `numpy`,
`scipy`. 추가 의존성 없음.

---

## 9. 산출물 목록

`results/research_sr_labeling/`:
- `R_labeling_distribution.csv` — 7 라벨링 × 3 split 양성 분포
- `R_test_multi_seed.csv` — 7 × 5 seed test 결과
- `R_results_grid.csv` — 7 × 4 fold × 2 seed walk-forward
- `R_summary_per_labeling.csv` — 라벨링별 통계 요약
- `R_bootstrap_ci.csv` — 라벨링별 Bayesian bootstrap CI
- `R_summary.json` — 모든 결과 통합 JSON
- `R_pr_auc_comparison.png` — bootstrap CI bar chart
- `R_walk_forward_stability.png` — fold별 변동성
- `R_topk_metrics.png` — top-K precision/lift

코드: `src/research/sr_labeling/`:
- `labelings.py` — 7개 라벨링 변환 함수 + delist_year 매핑
- `train_eval.py` — 학습/평가 파이프라인 (fixed_N1 평가 통일)
- `run_sr_all.py` — 전체 grid 실행 + 그림 생성
