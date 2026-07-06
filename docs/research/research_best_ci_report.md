# 천장 돌파의 통계적 입증

## Best baseline의 Bayesian Bootstrap CI vs S0 baseline의 paired 비교

작성일: 2026-05-31
작성자: 이현지
선행 연구: S0 진단 / S1 walk-forward / A1 감사 / A2 관리종목 / A3 시장
재현 코드: `src/research/best_ci/run_best_bootstrap.py`
결과 산출물: `results/research_best_ci/`

---

## 초록 (Abstract)

S0 진단에서 측정한 baseline RF의 test PR-AUC 95% Bayesian 신뢰구간은
**[0.180, 0.421]**이었다. A1 감사의견 추가까지 진행한 walk-forward 결과는
평균 0.408로 CI 상단(0.421) 근처에 도달했으나, 그것이 **통계적으로 명확
한 돌파인지** 확정하지 못한 상태였다 — walk-forward 평균은 단일 split이
아닌 4 fold 평균이라 baseline과 같은 조건이 아니었기 때문이다.

본 보고서는 그 질문에 대한 직접 답이다. **단일 split (train 2015~2022 /
valid 2023 / test 2024) — baseline과 정확히 같은 조건**에서:

1. 현재 best 모델 (**Transformer + 시장 + 감사, 44 features**)을 5 seed
   로 학습 → 5-seed test probability 평균
2. S0 canonical RF baseline을 동일 5 seed로 재학습 → 5-seed probability
   평균
3. 같은 Dirichlet(1,…,1) 가중치로 두 모델의 PR-AUC posterior를 **paired
   Bayesian bootstrap** (n=5,000) 으로 동시 산출
4. Δ = best − baseline posterior 분포 분석

세 가지 결과가 천장 돌파를 명확히 확정한다.

**(1)** Best 5-seed mean test PR-AUC는 **0.432 ± 0.026**, ensemble
probability point는 **0.454**, 95% Bayesian CI **[0.331, 0.586]**. **CI 하단
0.331이 baseline mean 0.270을 훨씬 넘는다**.

**(2)** Paired Δ posterior의 mean = **+0.183**, **95% CI [+0.094, +0.279]**
— **0을 전혀 포함하지 않는 순수 양수 영역**. 95% 신뢰수준에서 best는
baseline보다 최소 +0.094 PR-AUC만큼 우수함이 보장된다.

**(3)** **P(best > baseline) = 1.000** — 5,000개 bootstrap resample 중
baseline이 best를 이긴 케이스가 **단 한 번도 없음**.

종합: **"baseline의 0.2876 천장은 단단하지 않다"는 S0 진단의 명제가
완전히 확립**되었다. 단순 스냅샷 RF (PR-AUC 0.27) → 시계열 Transformer +
KRX 부실 신호 결합 (PR-AUC 0.43) 의 +0.16 PR-AUC 개선은 paired bootstrap
기준 **확실한 통계적 우위**다.

> **⚠ 사후 검증 (2026-06-20, `docs/research_rf_feature_control_report.md`)**
> 본 보고서의 +0.18 Δ는 **(a) 모델 (b) 시계열 입력 (c) 외부 피처** 세 변화를
> 동시에 포함한다. 이를 분리한 대조 실험 결과: 그 차이의 **약 80%는 외부 피처
> (감사·시장)** 에서 오며, 딥러닝·시계열 없이 **스냅샷 RF + 44 피처만으로도
> PR-AUC 0.413**에 도달한다. 외부 피처를 RF에도 동등하게 준 뒤 측정한 시계열의
> 순효과 Δ(T-44 − RF-44) = **+0.039, 95% CI [−0.040, +0.119] — 0과 구분되지
> 않는다**. 따라서 정확한 진술은 **"시계열 Transformer가 천장을 깼다"가 아니라
> "외부 부실 신호가 천장을 깼고, 시계열은 top-K 정밀도(P@20 0.55→0.75)를
> 더한다"** 이다. 아래 §4.1·§4.4·§9가 이 검증에 맞춰 수정되었다.

---

## 1. 배경: 왜 이 보고서가 필요한가

각 stream별 walk-forward 평균은 다음과 같이 진화했다.

| 모델 / 설정 | walk-forward 평균 PR-AUC | S0 baseline CI [0.18, 0.42] 안? |
|------------|--------------------------|----------------------------------|
| Transformer 33 features | 0.289 | 안에 |
| Transformer + 시장 (39) | 0.297 | 안에 |
| **Transformer + 시장 + 감사 (44)** | **0.408** | **상단 가장자리** |
| Transformer + 시장 + 감사 + 관리종목 (50) | 0.402 | 상단 가장자리 |

A1 보고서까지의 결론은 *"천장 돌파에 매우 근접"* 이었다. 그러나 다음
이유로 **확정적 결론을 내릴 수 없었다**:

1. **walk-forward 평균(0.408)은 train 데이터 크기가 다른 4 fold 평균** —
   baseline은 fold 4 (train 2015-2022) 단일 split에서 측정됨. 직접 비교
   부적절.
2. **단일 fold 4 walk-forward = 0.444였지만 3 seed만** 사용 — std 추정
   약함.
3. **baseline의 CI는 single seed bootstrap** — 5 seed로 다시 측정하면
   baseline도 약간 달라질 수 있음.

본 보고서는 이 모든 약점을 동시에 해결한다 — **같은 split, 같은 seed
구성, paired bootstrap**.

---

## 2. 방법론

### 2.1 데이터 / 모델

| 항목 | 값 |
|------|-----|
| Split | 단일 (train 2015~2022, valid 2023, test 2024) — baseline 동일 |
| Test 양성 | 56 / Test 음성 | 5,255 |
| Best 모델 | Transformer + Time2Vec (44 features: 33 financial + 6 market + 5 audit) |
| Baseline 모델 | RF (S0 canonical, 33 features) |
| Seeds | 42, 7, 13, 21, 100 |
| Best 모델 hyperparams | model_dim=64, heads=4, layers=1, dropout=0.2, lr=5e-4, focal γ=2 α=0.25, epochs=25 patience=8 |
| Baseline hyperparams | n_estimators=200, max_depth=10, min_samples_leaf=5, max_features=sqrt |

### 2.2 Paired Bayesian Bootstrap

두 모델 모두 **같은 test 행 인덱스**를 공유한다 (fixed_v1 test, 5,311 행).
따라서 같은 Dirichlet(1,…,1) 가중치 w_i ~ Dir(1) 를 5,000회 sampling하여
두 모델의 PR-AUC posterior를 동시 산출:

```
for i in 1..5000:
    w_i = Dirichlet(1, …, 1)        # n = 5,311 dim
    best_post[i]  = average_precision_score(y_test, best_avg_proba, sample_weight=w_i * n)
    base_post[i]  = average_precision_score(y_test, rf_avg_proba,   sample_weight=w_i * n)
```

Paired (같은 w_i)이라는 점이 중요. 두 모델의 차이가 단순한 모델 noise
인지 진짜 신호인지 직접 검증 가능.

### 2.3 결정 기준

- **Δ 95% CI 전부 양수** → 5% 양측 유의수준에서 best > baseline 입증
- **P(best > baseline) > 0.975** → 단측 2.5% 유의수준에서 best > baseline

---

## 3. 결과

### 3.1 5-seed PR-AUC 분포

| seed | Best (44 feat) | Baseline (33 feat, RF) |
|------|----------------|------------------------|
| 42 | 0.4617 | 0.2876 |
| 7 | 0.4465 | 0.2767 |
| 13 | 0.4250 | 0.2483 |
| 21 | 0.3845 | 0.2639 |
| 100 | 0.4406 | 0.2708 |
| **mean** | **0.432 ± 0.026** | **0.269 ± 0.013** |
| **ensemble point** | **0.454** | 0.270 |

### 3.2 Posterior 비교 (Bayesian bootstrap n=5,000)

| 항목 | Best | Baseline |
|------|------|----------|
| Ensemble point PR-AUC | 0.454 | 0.270 |
| Posterior mean | 0.453 | 0.279 (S0 측정치 0.294와 ±0.02 일치) |
| 95% CI | **[0.331, 0.586]** | [0.167, 0.402] |
| CI 폭 | 0.255 | 0.235 |

**핵심 관찰**: Best CI 하단 0.331이 baseline mean 0.270을 넘는다. 두
posterior는 거의 겹치지 않음.

### 3.3 Paired Δ posterior

| 항목 | 값 |
|------|-----|
| Δ mean | **+0.183** |
| Δ 95% CI | **[+0.094, +0.279]** |
| **P(best > baseline)** | **1.000** (5,000/5,000) |
| **Δ CI 전부 양수** | **✓ (0 미포함)** |

> 5,000번 paired bootstrap resample 중 best가 baseline을 이긴 경우 = 5,000
> 패배 경우 = 0

> 그림: `results/research_best_ci/best_ci_comparison.png`

---

## 4. 해석

### 4.1 무엇이 통계적으로 확정됐는가

1. ~~**S0 baseline의 95% CI [0.180, 0.421]을 완전히 돌파**. Best CI 하단 0.331 > baseline CI 상단 0.402 가까이.~~
   **Best CI 하단(0.331)이 baseline CI 상단(0.402)을 넘지는 않으나**, Paired Δ의 95% CI가 전부 양수([+0.094, +0.279])이므로 두 분포는 통계적으로 명확히 구별된다.
2. **Paired Δ CI 전부 양수**: 95% 신뢰수준에서 **최소 +0.094 PR-AUC**
   개선이 보장됨.
3. **P(best > baseline) = 1.000**: 통계적으로 baseline이 best를 이길 가능성이
   사실상 0.

### 4.2 +0.183 PR-AUC 차이를 어떻게 해석하는가

- 절대 차이 +0.183은 baseline의 PR-AUC 0.27 대비 **+68% 상대 향상**.
- top-20 운영 지표 (S1 walk-forward 보고서): 0.580 → 0.650 (+0.07).
  현재 단일 split에서는 더 클 것으로 예상.
- ROC-AUC: 0.855 → 0.919 (+0.064) — 양성 ranking 정밀도 큰 폭 개선.

### 4.3 무엇을 깬 것인가

S0 진단의 핵심 명제:
> "0.2876 천장"은 단단하지 않다. 평가 자체의 노이즈가 일부 기여하는
> 현상이다.

이 명제는 본 보고서에서 두 가지 의미로 확장된다:

1. **S0의 명제는 baseline의 CI 폭 0.24가 컸기 때문**. 즉 어떤 변화도
   noise에 묻힐 수 있는 환경이었다.
2. **그러나 진짜로 신호가 강한 변화 (시계열 + KRX 부실 신호)는 noise를
   훨씬 뛰어넘는다**. Best의 CI 하단 0.331이 baseline CI 상단 0.421을
   넘지는 않지만, paired Δ 분석은 두 분포의 차이가 분명함을 보여준다.

S0의 진단이 **"baseline이 단단해 보이는 천장이 아니라 noise band 안에
있다"**를 확립했다면, 본 보고서는 그 위에 **"진짜 강한 신호는 noise band
를 뛰어넘는다"**를 추가한다. 두 발견이 합쳐져 **방법론(통계적 엄밀성) +
결과(천장 돌파)** 라는 완성된 portfolio 형태가 된다.

### 4.4 best 모델의 핵심 구성요소 (2026-06-20 대조 실험으로 재귀속)

~~시계열 모델링 + A1 결합이 단순 스냅샷 RF에서 +68% 향상의 90%+를 설명한다~~ —
이 귀속은 모든 기여를 **Transformer 위에서만** 측정해 모델·시계열·피처를 분리
하지 못했다. `research_rf_feature_control`가 **동일 test 행 위에서 paired
bootstrap**으로 분리한 결과(단일 split, 5-seed ensemble):

| 분리된 효과 | paired Δ PR-AUC | 95% CI | 유의 |
|------------|------------------|--------|------|
| 외부 피처 전체 (스냅샷 RF-44 − RF-33) | **+0.156** | [+0.071, +0.245] | ✓ |
| └ 감사 단독 (RF+audit − RF-33) | +0.107 | [+0.040, +0.180] | ✓ |
| └ 시장 단독 (RF+market − RF-33) | +0.150 | [+0.055, +0.240] | ✓ |
| 시계열 순효과, 외부 피처 **無** (T-33 − RF-33) | +0.105 | [+0.031, +0.184] | ✓ |
| **시계열 순효과, 외부 피처 有 (T-44 − RF-44) ★** | **+0.039** | **[−0.040, +0.119]** | **✗** |
| 헤드라인 재현 (T-44 − RF-33) | +0.195 | [+0.104, +0.288] | ✓ |

세 가지 재귀속:
1. **천장 돌파의 1차 동력은 외부 부실 신호**다 — 딥러닝·시계열 없이 스냅샷 RF에
   44 피처만 줘도 PR-AUC **0.413**(헤드라인 +0.195의 80%). 감사 단독으로 +0.107.
2. **시계열과 외부 피처는 상호 대체재**다 — 시계열은 외부 피처가 *없을* 때
   +0.105로 유의하지만, *있으면* +0.039로 줄어 **0과 구분되지 않는다**(P=0.83).
   두 정보원이 같은 "부실 궤도" 신호를 중복 포착한다.
3. **그래도 시계열이 남기는 것은 top-K 정밀도**다 — PR-AUC가 동률인 지점에서도
   T-44의 P@20 **0.75** / P@50 **0.50** > RF-44의 0.55 / 0.44.

(주의: test 양성 56개라 +0.039를 검출할 검정력이 낮다. "비유의 = 효과 없음"이
아니라 "본 표본으로 확정 불가"이며, walk-forward paired 비교가 결정적 답을 준다.)

---

## 5. 한계 / 추가 보완 가능

- **단일 split** — fold 4 (가장 유리한 train data 양). walk-forward 평균
  (0.408)이 단일 split mean (0.432)보다 약간 낮음. 즉 본 보고서의 통계는
  fold 4 조건에서의 통계적 천장 돌파를 입증하며, walk-forward 평균의
  CI는 별도 측정 필요.
- **best 모델 hyperparameter 미튜닝** — model_dim=64 등 고정. Sweep 시
  추가 향상 가능성.
- **5 seed로 std 추정** — 10+ seed로 검증 시 std 정밀화.
- **paired bootstrap은 같은 weight 가정에 의존** — 두 모델이 정말 동일
  분포 평가에서 측정되는지 sanity check 완료 (test labels 일치 검증).
- **외부 valid 미사용** — 모든 통계는 in-sample. 향후 2025 데이터 도착
  시 진정한 out-of-sample 검증.
- **귀속(attribution) 한계** — 본 보고서의 Δ는 모델·시계열·외부 피처를 묶어서
  측정한다. `docs/research_rf_feature_control_report.md`(2026-06-20)가 이를
  분리해, 돌파의 80%가 외부 피처에서 오고 시계열의 순 PR-AUC 기여는 외부 피처
  동등 조건에서 비유의(+0.039)임을 보였다. **4-fold walk-forward paired 비교
  (측정 4배)에서도 동일 — Δ_ts_full = +0.021, 95% CI [−0.062, +0.105], P=0.69.
  RF-44의 walk-forward 평균 0.399 ≈ 본 best 모델의 0.408**(같은 천장에 시계열
  없이 도달). **단 이는 시장 피처를 포함한 조건이다.** #8(`research_market_check
  _report.md`)은 그 시장 피처가 양성의 45%를 차지하는 "frozen(동결주) 탐지"
  artifact임을 밝혔고, #8b(`research_frozen_check_report.md`)는 **시장 피처를
  제거하면(38=재무+감사) 시계열이 스냅샷을 Δ=+0.090, CI [+0.029, +0.158],
  P=0.998로 유의하게 능가**함을 보였다. 즉 시계열의 PR-AUC 가치는 실재하며,
  시장 artifact가 스냅샷 RF에 crutch를 줘 가렸던 것이다. **정정된 best 모델 =
  Transformer + 재무 + 감사 (38 features, 시장 제외) = 0.458** (본 보고서의 44feat
  0.454보다 높고 더 단순). 천장 돌파의 정당한 동력은 **감사 피처 + 시계열
  모델링**이며, 시장 피처는 폐기 권장.

---

## 6. 다음 단계 권장

본 stream의 핵심 결과는 확립됨. 후속 가능 방향:

| 항목 | 가치 | 비용 |
|------|------|------|
| **Walk-forward 4 fold × 5 seed + bootstrap** | walk-forward 평균의 통계적 CI 확립 | ~2시간 |
| **Hyperparameter sweep** | 추가 +0.01~0.03 가능 | 4~6시간 |
| **Stream 3 hazard 모형** (Cox / Discrete hazard) | fixed-N 의존성 제거 + 학술적 차별성 | 1~2일 |
| **Phase 4 Neural CDE** | UNIST DAL 시그니처 | GPU + 1주 |
| **2025 데이터 도착 시 OoS 검증** | 진짜 generalization | 2025 데이터 대기 |
| **Portfolio 정리** | 현재 시점에서 진행 가능 | 1~2일 |

---

## 7. 재현성

```powershell
.venv\Scripts\python.exe -m src.research.best_ci.run_best_bootstrap
```

소요 시간: ~30분 (Transformer 5 seed + RF 5 seed + bootstrap 5,000).
모든 seed/bootstrap 고정. 동일 seed 재실행 시 결과 일치.

---

## 8. 산출물

`src/research/best_ci/run_best_bootstrap.py` — paired Bayesian bootstrap 비교.

`results/research_best_ci/`:
- `best_ci_summary.json` — 통합 결과 (5-seed PR-AUC, posterior CI, Δ posterior)
- `seed_results.csv` — seed별 best vs baseline PR-AUC
- `best_ci_comparison.png` — posterior overlay + Δ posterior

---

## 9. 핵심 한 줄

> **단순 스냅샷 RF (0.27) → 풀 모델 (0.45). paired bootstrap 5,000/5,000 모두
> 우위 → 천장은 통계적으로 깨졌다. 단, 대조 실험(2026-06-20)에 따르면 그 돌파의
> 1차 동력은 시계열이 아니라 외부 부실 신호(감사·시장)이며 — 스냅샷 RF + 44
> 피처만으로 0.413 — 시계열의 추가 PR-AUC(+0.039)는 외부 피처 동등 조건에서
> 통계적으로 확정되지 않는다. 시계열의 확정된 가치는 top-K 정밀도(P@20
> 0.55→0.75)에 있다.**
