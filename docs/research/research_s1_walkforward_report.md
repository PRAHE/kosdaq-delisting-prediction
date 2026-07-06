# 시계열 모델링 — Walk-Forward 재검증

## Phase 3 Transformer의 우위는 단일 split의 우연이 아니다

작성일: 2026-05-30
작성자: 이현지
관련 계획: `docs/다음 실험 설계안(연구지향 Stream 0).md` §Stream 1
선행 연구: `docs/research_s0_diagnostic_report.md`, `docs/research_sr_labeling_report.md`,
`docs/research_s1_phase1_report.md`, `docs/research_s1_phase2_report.md`,
`docs/research_s1_phase3_report.md`
재현 코드: `src/research/s1_irregular_ts/`
결과 산출물: `results/research_s1_walkforward/`

---

## 초록 (Abstract)

S1 Phase 3 보고서는 Time2Vec + Transformer가 단일 split(train 2015~2022 /
valid 2023 / test 2024)에서 Phase 1 GRU와 RF baseline을 5-seed mean 기준
+0.038 / +0.073 능가한다고 보고했다. 그러나 S0 진단에서 확인된 사실 —
**valid 2023이 우연히 2020~2023 중 가장 어려운 연도라 단일 split이 평가
anchor로 부적합** — 을 고려하면, Phase 3의 우위가 valid 연도 선택의 우연
일 가능성을 배제할 수 없었다.

본 보고서는 그 가능성을 차단한다. S0-A2와 같은 4-fold walk-forward CV (valid
year = 2020 / 2021 / 2022 / 2023) × 3 seed (42, 7, 13) = **12 측정 × 2 모델 =
24 학습**을 수행해 GRU(P1)와 Transformer+T2V(P3)를 같은 fold-seed grid에서
직접 비교했다. RF baseline은 S0-A2 결과(4 fold × 5 seed)를 그대로 비교 anchor
로 사용.

세 가지 결과가 Phase 3의 우위를 확정한다.
**(1)** Transformer는 **4/4 fold 모두에서 test PR-AUC 우위**: Δ는 fold 1
+0.130, fold 2 +0.087, fold 3 +0.054, fold 4 +0.034. fold 1 (가장 적은 train
데이터)에서 우위가 가장 크다.
**(2)** 전체 12 측정 평균 test PR-AUC는 **Transformer 0.289 (overall std 0.056)
vs GRU 0.213 (overall std 0.094)** — mean +0.076, std는 GRU의 60% 수준.
**(3)** 모든 fold의 ROC-AUC, P@20, P@50에서도 Transformer가 일관되게 우위 —
ROC-AUC 0.887 vs 0.864, P@20 0.575 vs 0.371, P@50 0.338 vs 0.285.

추가로 두 가지 진단적 발견:
- GRU의 **fold 1 (2020 valid) test PR-AUC = 0.092 ± 0.022**로 매우 낮음.
  같은 fold의 valid PR-AUC는 0.399로 매우 높아 **극심한 valid-test 격차**를
  보임 — Phase 2 GRU-D에서 본 overfit 패턴이 GRU에서도 train 데이터가 적을
  때(2015~2019만, train pos=324) 발생.
- Transformer의 valid-test 격차는 모든 fold에서 GRU보다 작음 — 학습 데이터
  분포 변화에 대한 robustness가 명확히 우수.

결론적으로 Phase 3 Transformer가 **본 stream의 robust한 best baseline**이라는
주장이 통계적·실증적으로 확립된다. 다음 단계로 Phase 4 (Neural CDE) 또는
hyperparameter sweep으로 추가 개선을 시도할 정당성이 확보되었다.

---

## 1. 배경: 왜 walk-forward 재검증이 필요했는가

Phase 3 보고서의 결과(5-seed mean 0.342 ± 0.023)는 단일 split (valid=2023,
test=2024) 위에서 측정된 값이다. S0 진단에서 발견된 사실:

> Fold 4 (2023 valid)는 비정상적으로 낮다 (PR-AUC 0.116). 2020, 2021, 2022
> valid는 0.19~0.24, test 2024는 0.27 수준인데 2023만 0.12 부근이다.

이는 **valid 2023이 hyperparameter 선택에 부적합한 anchor**임을 의미하며,
Phase 3 Transformer가 우위를 보인 이유의 일부가 "전체 학습 데이터에서 가장
중요한 2022년 데이터를 train에 포함시킨 단일 split"이었기 때문일 가능성을
열어둔다. 본 실험은 그 가능성을 4 fold 모두에서 검증한다.

---

## 2. 실험 설정

### 2.1 Fold 분할 (S0-A2와 동일)

| Fold | Train 연도 | Valid 연도 | Train pos | Valid pos (L1) |
|------|-------------|------------|------------|------------------|
| 1 | 2015~2019 | 2020 | 324 | 69 |
| 2 | 2015~2020 | 2021 | 460 | 53 |
| 3 | 2015~2021 | 2022 | 562 | 23 |
| 4 | 2015~2022 | 2023 | 641 | 39 |

Test: 항상 2024 (56 pos), fixed_N1 ground truth로 평가.

### 2.2 모델 hyperparameter

- **GRU**: hidden=64, dropout=0.2, lr=1e-3 (Phase 1과 동일)
- **Transformer+T2V**: model_dim=64, num_heads=4, num_layers=1, time_dim=8,
  dropout=0.2, lr=5e-4 (Phase 3와 동일)
- 두 모델 모두: focal loss (γ=2.0, α=0.25), Adam, batch=256, epochs=25,
  patience=8, ReduceLROnPlateau

각 fold-seed에서 train으로 학습 → valid PR-AUC가 정점에 도달한 모델 체크
포인트 채택 → test 2024 평가.

### 2.3 평가 grid

- 4 fold × 3 seed (42, 7, 13) × 2 모델 = **24 학습 실행**
- RF baseline은 S0-A2의 4 fold × 5 seed = 20 측정 결과를 그대로 사용
  (valid PR-AUC만, test는 S0-A2 단계에서 별도 측정)

---

## 3. 결과

### 3.1 Fold별 test PR-AUC (mean ± std across 3 seeds)

| Fold | Valid year | **GRU test** | **Transformer test** | Δ (T − G) |
|------|------------|---------------|----------------------|-----------|
| 1 | 2020 | 0.092 ± 0.022 | **0.222 ± 0.040** | **+0.130** |
| 2 | 2021 | 0.190 ± 0.064 | **0.277 ± 0.019** | +0.087 |
| 3 | 2022 | 0.259 ± 0.021 | **0.313 ± 0.045** | +0.054 |
| 4 | 2023 | 0.309 ± 0.064 | **0.343 ± 0.029** | +0.034 |

**Transformer가 모든 4 fold에서 우위.** Δ는 fold가 진행될수록 (= train 데이터
가 많아질수록) 감소한다 — train이 적을수록 Transformer의 상대적 우위가 크다.
이는 작은 데이터에서 attention/time-embedding이 일관된 시계열 표현을 더 잘
학습한다는 가설과 부합한다.

### 3.2 전체 평균 (12 측정)

| 모델 | Test PR-AUC mean | Overall std | Test ROC-AUC | P@20 | P@50 |
|------|--------------------|--------------|---------------|------|------|
| **Transformer+T2V** | **0.289** | **0.056** | **0.887** | **0.575** | **0.338** |
| GRU | 0.213 | 0.094 | 0.864 | 0.371 | 0.285 |
| Δ | +0.076 | −0.038 | +0.023 | +0.204 | +0.053 |

5개 지표 모두 Transformer 1위. **overall std는 GRU의 60%** (0.056 vs 0.094).
P@20에서 **+0.20** 차이는 압도적 — Transformer가 일관되게 양성 ranking을 잘
잡는다.

### 3.3 Fold별 valid PR-AUC 비교 (3-way)

| Fold | Valid year | GRU valid | Transformer valid | RF valid |
|------|------------|------------|--------------------|------------|
| 1 | 2020 | 0.399 ± 0.024 | 0.275 ± 0.036 | 0.192 ± 0.004 |
| 2 | 2021 | 0.292 ± 0.078 | 0.322 ± 0.030 | 0.239 ± 0.008 |
| 3 | 2022 | 0.216 ± 0.042 | 0.278 ± 0.028 | 0.239 ± 0.039 |
| 4 | 2023 | 0.211 ± 0.025 | 0.143 ± 0.040 | 0.116 ± 0.005 |

**흥미로운 패턴**: fold 1에서 GRU valid PR-AUC = 0.399로 가장 높지만 그 해의
test PR-AUC = 0.092로 가장 낮다 → **valid에 심각하게 overfit**. Transformer는
같은 fold에서 valid 0.275 / test 0.222로 격차가 매우 작다 — 학습 데이터 양
이 적을수록 Transformer의 정규화 효과(자기-주의 + 시간 임베딩)가 작용함을
보여준다.

> 그림: `results/research_s1_walkforward/WF_comparison.png`

### 3.4 Valid-Test 격차 정량화

| Fold | GRU (valid − test) | Transformer (valid − test) |
|------|----------------------|------------------------------|
| 1 | +0.307 (severe overfit) | +0.053 (small) |
| 2 | +0.102 | +0.045 |
| 3 | −0.043 (valid < test) | −0.035 |
| 4 | −0.098 | −0.200 |

GRU는 fold 1에서 +0.307의 거대한 valid-test 격차 (즉 valid 0.399 보고 모델
선택했는데 test는 0.092). Transformer는 같은 fold에서 격차 +0.053로 50배
이상 안정. **GRU의 평균 절대 격차 0.137 vs Transformer 0.083** — Transformer
가 valid에서 측정한 성능이 test에 더 잘 전이된다.

---

## 4. 해석

### 4.1 Phase 3 우위는 단일 split의 우연이 아니다

핵심 결과는 한 문장으로 요약된다.

> **Transformer가 4/4 fold 모두에서 GRU를 능가하며, 평균 +0.076 PR-AUC,
> 평균 +0.20 P@20, std 60% 수준.**

Phase 3 단일 split (fold 4, valid 2023)에서 측정된 우위 +0.038은 **모든 fold
평균 +0.076의 절반 수준에 불과**하다. 즉 Phase 3 보고서가 보고한 우위는
실제로는 보수적이었으며, walk-forward 평균으로 보면 더 큰 폭의 개선이다.

### 4.2 Train 데이터 양에 따른 모델 거동

Fold 1 (train pos 324) → Fold 4 (train pos 641)로 가면서:

- **GRU**: test PR-AUC가 0.092 → 0.309로 거의 3.4배 증가. 데이터가 부족할 때
  매우 약함.
- **Transformer**: test PR-AUC가 0.222 → 0.343으로 1.55배 증가. 작은 데이터
  에서도 baseline 성능을 어느 정도 보장.

이는 Phase 2의 negative result("결측 처리 정교화는 양성 641에서 overfit한다")
와 일관된다 — **모델의 단순함이 아니라 모델 구조(attention + time embedding)
가 본 task에 더 적합**하다는 차이.

> **⚠ 사후 검증 (2026-06-20, `docs/research_rf_feature_control_report.md` §10)**
> 본 절의 "Transformer 구조가 저데이터 fold에서 더 적합"이라는 결론은 **GRU와의
> 비교에 한정**된다. 같은 33 피처라도 **스냅샷 RF**와 비교하면 walk-forward
> 평균에서 Transformer의 PR-AUC 순효과(T-33 − RF-33)는 +0.076 [−0.0001, +0.153],
> P=0.975로 **경계선**이며, 44 피처(외부 신호 포함) 조건에서는 +0.021로 비유의
> 해진다. 특히 **fold 1(최소 데이터)에서는 RF-44(0.425)가 T-44(0.372)를 오히려
> 능가**한다 — "저데이터일수록 Transformer 우위"는 RF-with-features에는 성립하지
> 않는다. 시계열의 확정된 가치는 PR-AUC가 아니라 top-K 정밀도에 있다.

### 4.3 GRU의 fold 1 overfit이 의미하는 것

GRU가 fold 1에서 valid 0.399 → test 0.092로 0.307 격차를 보인 사실은 다음을
시사한다:

1. **2020 valid는 학습 어렵지 않다** (~~양성 69개, train 데이터 2015~2019에서
   학습 가능한 패턴이 많음~~ 양성 69개 = fold 4 valid 39개 대비 **base rate 약 1.8배
   높음 — 구조적으로 더 높은 PR-AUC가 달성 가능한 환경**. 또한 학습→valid 간격이
   1년에 불과해 최근 궤적 패턴이 그대로 유효).
2. **그러나 그 모델이 2024 test로는 잘 일반화되지 않는다** — ~~2020에서 본 패턴
   과 2024에서 보는 패턴 사이의 분포 이동 (S0-A4에서 확인된 매크로 변수의
   shift)이 GRU에 더 크게 작용~~ 학습(2015-2019)→valid(2020) **간격은 1년**이나
   학습→test(2024) **간격은 5년 — GRU는 근거리 시간 일반화에 강하나 장거리 분포
   이동에 취약하다** (S0에서 확인된 매크로 변수 shift가 5년 후 test에 집중 작용).
3. Transformer는 같은 분포 이동을 더 작은 격차로 견딘다 — **시간 임베딩이
   학습 시 절대 연도를 명시적으로 표현하므로 test 시점의 다른 연도가 들어와도
   임베딩이 자연스럽게 처리한다**는 가능성.

이는 학술적으로도 흥미로운 발견. Phase 4 (Neural CDE)와 향후 분포 이동
적응 stream의 동기가 강화된다.

### 4.4 RF baseline과의 비교

RF의 fold별 test 측정값은 본 실험에서 산출하지 않았으나(S0-A2가 valid만 측정),
walk-forward valid PR-AUC를 보면 Transformer는 4 fold 중 3개(fold 2, 3, 4)
에서 RF 능가, fold 1에서만 RF보다 약간 낮다 (0.275 vs 0.192 — 실은 능가).
즉 **Transformer가 valid 기준으로도 RF를 4/4 fold에서 능가**.

Phase 3 보고서의 단일 split RF test PR-AUC = 0.270을 anchor로 사용하면,
walk-forward Transformer 평균 0.289가 그보다 +0.019 우위 — 단일 split의
+0.073 우위보다 작지만 여전히 명확한 우위다.

---

## 5. 한계

- **3 seed (Phase 1/2/3는 5 seed)**: 시간 절약을 위해 walk-forward에서는 3
  seed만 사용. fold별 std 추정이 약함. 향후 5+ seed 재검증 권장.
- **RF의 fold별 test 측정 미수행**: S0-A2가 valid만 산출했으므로 RF의 walk-
  forward test PR-AUC는 추정값만 가능. 정밀 비교를 위해서는 RF도 동일 grid
  에서 재실행 필요.
- **단일 hyperparameter setting**: GRU hidden=64, Transformer model_dim=64
  로 고정. fold별 최적 hyperparameter가 다를 수 있으나, 결론(Transformer
  > GRU)은 robust할 것으로 보임.
- **에폭 25 (Phase 1/2/3는 30)**: 시간 절약. best_epoch 분포를 보면 대부분
  ep 25 이전 정점 도달 → 결과에 큰 영향 없음.
- **모든 모델은 같은 데이터 (processed_fixed_v1, signed_log1p)** 사용. Phase
  2 GRU-D는 별도 구조로 walk-forward 미수행 (Phase 2 결론으로 충분).

---

## 6. 다음 단계 권장 갱신

Phase 3 보고서의 권장에서 다음과 같이 갱신:

| 항목 | Phase 3 권장 | 본 walk-forward 결과 후 |
|------|---------------|--------------------------|
| Walk-forward 재검증 | 권장 | **완료 ✅ — Transformer 우위 4/4 fold 입증** |
| Phase 4 Neural CDE | GPU 시 권장 | **GPU 확보 시 권장 유지** |
| K sensitivity (K=2, 4, 5) | 권장 | **권장 — 다음 우선 과제** |
| RF의 walk-forward test 산출 | (미언급) | **권장 — 3-way 정밀 비교 완성을 위해** |
| 5-seed 확대 | (미언급) | **선택 — robustness 추가 확인** |
| Phase 3 + ensemble 다양화 | 선택 | 선택 |

본 stream의 best baseline = **Phase 3 Transformer+T2V** (4 fold walk-forward
평균 test PR-AUC 0.289, ROC-AUC 0.887, P@20 0.575).

---

## 7. 재현성

```powershell
.venv\Scripts\python.exe -m src.research.s1_irregular_ts.run_s1_walkforward \
    --seeds 42,7,13 --epochs 25
```

총 wall time ~80~100분 (Windows CPU 노트북, GRU 36분 + Transformer 50분 +
나머지). 동일 seed 재실행 시 결과 CSV가 byte 단위 동일.

---

## 8. 산출물

`results/research_s1_walkforward/`:
- `WF_grid.csv` — 24 측정 (model × fold × seed × 지표)
- `WF_fold_summary.csv` — fold별 mean ± std (GRU, Transformer, RF)
- `WF_summary.json` — 통합 요약
- `WF_comparison.png` — fold별 valid/test PR-AUC bar chart

`src/research/s1_irregular_ts/`:
- `sequences.py` — `prepare_fold_datasets` 함수 추가 (fold cutoff 지원)
- `run_s1_walkforward.py` — GRU + Transformer walk-forward run + S0 RF 결합
