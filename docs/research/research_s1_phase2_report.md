# 시계열 모델링 — Phase 2: GRU-D

## 결측을 명시적으로 모델링하면 단순 GRU를 이기는가?

작성일: 2026-05-30
작성자: 이현지
관련 계획: `docs/다음 실험 설계안(연구지향 Stream 0).md` §Stream 1
선행 연구: `docs/research_s0_diagnostic_report.md`, `docs/research_sr_labeling_report.md`,
`docs/research_s1_phase1_report.md`
재현 코드: `src/research/s1_irregular_ts/`
결과 산출물: `results/research_s1_phase2/`

---

## 초록 (Abstract)

Phase 1 GRU는 단일 스냅샷 RF 대비 PR-AUC ensemble +0.106의 개선을 보였으나,
seed-variance가 RF의 3배에 달했다 (0.046 vs 0.015). Phase 2는 그 다음 자연스러운 가설을 검증한다: **feature-level missing을 GRU-D (Che et al. 2018)로
명시적으로 모델링하면 학습 안정성과 성능이 함께 개선되는가**.

데이터 측면에서 Phase 1과 가장 다른 점은 `processed_fixed_v1`에서 시작한
이미-imputed 데이터를 버리고, **`combined_raw.csv`에서 시작해 NaN을 보존**
한다는 것이다 — 그 결과 train 전체에서 약 24% feature-timestep이 결측
(예: 상각비 항목 70~75% 결측). 거시변수 6개는 fixed_v1에서 join하여 같은
33 피처를 유지. GRU-D는 feature별 input decay (`γ^x_d`) 와 hidden decay
(`γ^h`) 를 학습하여 결측 위치를 last-observed × γ + mean × (1-γ) 로 합성한다.

다섯 가지 결과가 **명료한 negative result**를 만든다.
**(1)** GRU-D의 5-seed 평균 test PR-AUC는 **0.267 ± 0.068**로 Phase 1
GRU의 0.304 ± 0.046보다 평균 0.037 낮고, std는 1.5배 크다.
**(2)** 그러나 GRU-D의 **ROC-AUC 0.875**는 Phase 1 GRU의 0.853과 RF의
0.855보다 모두 높다 — *ranking 자체*는 GRU-D가 가장 잘하지만 **precision-
recall 곡선의 양성 영역**에서 손해를 본다.
**(3)** Ensemble bootstrap PR-AUC는 GRU-D 0.315 [0.208, 0.442], GRU 0.376
[0.257, 0.513], RF 0.270 [0.166, 0.408] — GRU-D가 RF는 능가하나 GRU에는
못 미친다.
**(4)** Top-20 precision은 GRU-D=GRU=0.56으로 동률, Top-50은 GRU-D 0.328로
GRU 0.380보다 낮다.
**(5)** 학습 곡선에서 GRU-D는 valid PR-AUC가 epochs 25~27까지 상승하며
조기 종료가 거의 일어나지 않는다 (Phase 1은 ep 13~23). 동시에 test PR-AUC
는 epoch 15 부근에서 정점 도달 — **고전적인 overfit 패턴**.

원인은 분명하다. GRU-D는 GRU 대비 추가 파라미터(W_dec_x ∈ R^33, b_dec_x ∈
R^33, W_dec_h ∈ R^{33×64}) 약 2,200개를 가진다. 양성 641개·valid 양성 39개
환경에서는 이 추가 자유도가 학습 신호보다 noise를 더 학습한다. **결측 처리
의 명시적 모델링은 본 데이터 규모에서 가성비가 나쁘다**.

결론적으로 Phase 1 GRU가 본 stream의 사실상 best baseline 위치를 유지한다.
Phase 3 (Transformer + time2vec)와 Phase 4 (Neural CDE) 는 GRU-D의
"feature-level decay"보다 **시계열 표현 자체의 다양화**에 무게가 실리므로
여전히 정당화된다.

---

## 1. 배경: Phase 1이 남긴 두 질문

Phase 1 GRU는 두 가지 한계를 남겼다.

1. **seed-variance**: 0.046 (RF의 3배). 단일 seed에서는 0.265~0.382로 큰 폭
   변동. 학습 안정성 개선이 시급.
2. **결측 정보의 부재**: Phase 1은 `processed_fixed_v1`의 이미-imputed 데이터
   를 사용. 즉 어느 feature가 어느 timestep에 결측이었는지 정보가 사라진 상태
   였다. 결측 위치 자체가 신호일 수 있다는 가설이 검증되지 않음.

GRU-D (Che et al. 2018, *Scientific Reports*) 는 두 문제를 동시에 다룬다.

- Input decay `γ^x_t = exp(-ReLU(W_γx · δ_t + b_γx))`: feature별 시간 감쇠
- Hidden decay `γ^h_t = exp(-ReLU(W_γh · δ_t + b_γh))`: 결측 동안 hidden
  state도 감쇠
- Imputation `x̂_t = m_t · x_t + (1 - m_t) · (γ^x_t · x_last + (1-γ^x_t) · x̄)`
- GRU 입력: `[x̂_t ; m_t]` (mask 자체를 feature로 concat)

이론적으로 GRU-D는 결측 패턴이 신호와 무관하지 않을 때(MAR/MNAR) 단순 GRU
+ imputation을 능가한다. **본 phase는 우리 데이터가 그런지 검증한다**.

---

## 2. 실험 설정

### 2.1 데이터 — 결측 보존 시퀀스

| 항목 | Phase 1 | Phase 2 |
|------|---------|---------|
| Raw 데이터 | `processed_fixed_v1` (이미 imputed) | `combined_raw.csv` (NaN 보존) |
| 거시변수 6개 | 포함 | fixed_v1에서 join하여 포함 |
| 결측 처리 | SimpleImputer(median) | 모델 내부 GRU-D imputation |
| 시퀀스 mask | timestep-level (B,K) | **feature-level (B,K,D)** |
| time gap δ | 없음 | feature별 결측거리 (B,K,D) |

| Split | N | K | D | obs_ratio | 양성 (L3 → L1) |
|-------|---|---|---|-----------|----------------|
| train | 29,986 | 3 | 33 | 0.756 | 641 (L3) |
| valid | 5,050  | 3 | 33 | 0.846 | 39 (L1) |
| test  | 5,311  | 3 | 33 | 0.847 | 56 (L1) |

가장 결측 많은 feature 3개:
- 무형자산상각비 (관측률 0.24)
- 유형자산상각비 (관측률 0.26)
- 감가상각비 (관측률 0.27)

대부분 회사가 해당 항목을 공시하지 않음. 가장 결측 적은 3개는 거시변수 (관측률
~0.83~0.94). 신호 강한 재무비율(현금비율 등)은 95% 이상 관측됨.

### 2.2 모델

```
Input (B, K=3, D=33), Mask (B, K, D), X_last (B, K, D), Delta (B, K, D)

for k in 1..K:
    γ^x_k = exp(-ReLU(W_dec_x · δ_k + b_dec_x))   # (B, D), element-wise
    x̂_k   = m_k * x_k + (1 - m_k) * (γ^x_k * X_last_k + (1 - γ^x_k) * x̄)
    γ^h_k = exp(-ReLU(W_dec_h(δ_k)))              # (B, hidden)
    h_{k-1} ← γ^h_k * h_{k-1}                     # hidden decay
    h_k = GRUCell([x̂_k ; m_k], h_{k-1})           # 2D input
head = MLP(h_K) → sigmoid
```

- Trainable parameters: GRU 부분 동일 + (W_dec_x 33, b_dec_x 33, W_dec_h
  33×64+64=2,176) = **+2,242개 추가** (vs Phase 1).
- Loss: focal (γ=2, α=0.25), optimizer/scheduler/patience Phase 1과 동일.
- Scaling: `signed_log1p` 적용 (Phase 1과 일관).

### 2.3 평가

- 5 seed × 1 split, 동일한 fixed_N1 valid/test ground truth.
- 5-seed ensemble bootstrap CI (n=2000).
- 비교: Phase 1 GRU 5-seed, S0 canonical RF 5-seed.

---

## 3. 결과

### 3.1 5-seed 측정

| seed | best_ep | valid PR | **test PR** | test ROC | P@20 | P@50 |
|------|---------|----------|--------------|----------|------|------|
| 42   | 25      | 0.229    | 0.312        | 0.890    | 0.55 | 0.36 |
| 7    | 12      | 0.209    | **0.353**    | 0.866    | 0.75 | 0.40 |
| 13   | 27      | 0.234    | 0.262        | 0.880    | 0.55 | 0.34 |
| 21   | 27      | 0.256    | 0.218        | 0.867    | 0.55 | 0.28 |
| 100  | 27      | 0.302    | 0.187        | 0.871    | 0.40 | 0.26 |

### 3.2 모델 간 비교 (5-seed × ensemble bootstrap)

| 모델 | 5-seed mean ± std | Ensemble point | 95% CI | ROC-AUC | P@20 | P@50 |
|------|---------------------|------------------|---------|---------|------|------|
| **GRU-D (Phase 2)** | 0.267 ± 0.068 | 0.315 | [0.208, 0.442] | **0.875** | 0.560 | 0.328 |
| GRU (Phase 1) | **0.304 ± 0.046** | **0.376** | **[0.257, 0.513]** | 0.853 | 0.560 | **0.380** |
| RF baseline (S0) | 0.269 ± 0.015 | 0.270 | [0.166, 0.408] | 0.855 | **0.580** | 0.368 |

> 그림: `results/research_s1_phase2/P2_pr_auc_comparison.png`

### 3.3 학습 곡선 관찰

`results/research_s1_phase2/P2_train_curves.png`:

- best_epoch이 5 seed 중 4개에서 25 이상 (Phase 1은 13~23).
- valid PR-AUC가 epoch 27까지 상승하지만 그 시점의 test PR-AUC는 모두 하락
  → **고전적인 train-valid-test gap (overfit)**.
- Phase 1 quick run (epochs=15)에서 seed 42가 test PR-AUC **0.377**을 기록했
  지만 full run (epochs=30) 에서 0.312로 떨어진 사실이 이 패턴을 직접 보여줌.

---

## 4. 해석

### 4.1 핵심: feature-level missing 모델링이 도움이 안 됐다

GRU-D의 추가 파라미터 ~2,242개는 양성 641개(train) / 39개(valid) 환경에서는
**학습 신호보다 noise를 더 학습**하는 듯하다. 구체적으로:
- valid가 작아 hyperparameter 선택이 흔들림 → best_epoch이 평균 23.6
- 그 시점의 test PR-AUC가 평균 0.267로 best가 아님
- seed 100은 valid 0.302까지 가지만 test 0.187 — 완전한 overfit 신호

본 데이터에서 결측 패턴 자체가 신호와 강한 상관이 없거나, 있더라도 추가 파라
미터로 학습할 만큼 일관되지 않은 것으로 해석된다.

### 4.2 그러나 GRU-D가 더 잘하는 한 가지: ROC-AUC

| 모델 | ROC-AUC | PR-AUC |
|------|---------|---------|
| GRU-D | **0.875** | 0.267 |
| GRU   | 0.853 | 0.304 |
| RF    | 0.855 | 0.269 |

GRU-D가 ROC-AUC 1위. 이는 GRU-D가 **전체 ranking은 더 잘 만든다**는 의미다.
다만 precision-recall 곡선의 양성 영역(우리가 관심 있는 영역)에서 손해를 본다.
이 현상은 두 가지 가능성을 시사한다:
- GRU-D가 *전반적 ranking*에는 도움이 되나 *상위 양성*을 끌어올리는 데는
  약하다.
- 또는 calibration이 GRU-D 쪽이 더 좁아져 양성 확률이 평탄해진다.

후속 분석 (calibration plot)을 통해 추가 점검 가치가 있다.

### 4.3 5-seed mean과 ensemble의 격차

- GRU-D: mean 0.267 → ensemble 0.315 (+0.048)
- GRU:    mean 0.304 → ensemble 0.376 (+0.072)

두 모델 모두 ensemble로 성능 향상되지만 GRU 쪽이 더 큰 폭으로 이득.
GRU-D의 5-seed 예측 확률이 더 강하게 상관되어 있다 (다양성이 적다) → 평균화
의 이득이 작다. Phase 1 GRU의 단순함이 오히려 **bias-variance trade-off에서
sweet spot에 더 가까이** 위치한다.

### 4.4 negative result의 가치

본 phase의 결과는 "한 줄로 본 stream의 가설을 결정한다":

> 본 데이터 규모(641 train pos, 39 valid pos)에서는 단순 GRU + median
> imputation이 GRU-D + 결측 모델링보다 일관되게 우수하다.

이 발견은 후속 단계(Phase 3 Transformer, Phase 4 NCDE)의 설계에 직접 반영
된다 — **모델 복잡도를 늘리는 방향보다 시계열 표현을 다양화하거나 정규화를
강화하는 방향**이 더 유망하다.

---

## 5. 한계

- **단일 split** (walk-forward CV 미수행). 2023 valid의 본질적 어려움이 결과
  의 절대값에 영향. fold CV에서는 GRU vs GRU-D 우열이 바뀔 가능성 있음.
- **K=3 고정**. K=4, 5에서 GRU-D의 long-range decay 효과가 더 드러날 가능성.
- **Hyperparameter sweep 없음** (hidden=64 고정). GRU-D는 더 작은 hidden_dim
  으로 overfit 완화 가능성. 그러나 본 phase의 결론(추가 파라미터가 도움 안
  됨)은 sweep의 어느 점에서도 큰 폭으로 바뀌지 않을 것으로 보인다.
- **결측 패턴 분석 부족**. 상각비 3개 feature가 가장 결측이 많은데, 이 feature
  들의 결측 자체가 "기업 규모/유형" 신호일 가능성. 추가 분석 가치.
- **5-seed로는 std 추정이 약함**. Phase 1과 동일.

---

## 6. 다음 단계 (Phase 3, 4 권장 갱신)

| Phase | 원래 계획 | Phase 2 결과 후 |
|-------|-----------|-----------------|
| **Phase 3** Transformer + time2vec | 권장 | **권장 유지** — self-attention은 K=3에서도 의미 있는 비교 |
| **Phase 4** Neural CDE | 선택 | **권장** — irregular sampling을 진정으로 활용 (단 GPU 필요) |
| **추가** Phase 1 GRU의 정규화 강화 | - | **신규 권장** — dropout↑, hidden↓, label smoothing 등으로 seed-variance 줄이기 |

Phase 1 GRU가 본 stream의 *현재까지의 best simple baseline*이며, 후속 phase
는 GRU를 깊게 능가하기 위해 **시계열 표현 자체를 다양화**하는 방향이 더
유망하다는 것이 본 phase의 결론이다.

---

## 7. 재현성

```powershell
# 단일 seed (~3분)
.venv\Scripts\python.exe -m src.research.s1_irregular_ts.run_s1_phase2 --quick

# 5-seed full (~40~50분 CPU)
.venv\Scripts\python.exe -m src.research.s1_irregular_ts.run_s1_phase2
```

Hyperparameter (default): hidden=64, dropout=0.2, focal γ=2 α=0.25, lr=1e-3,
batch=256, epochs=30, patience=8, labeling=L3_rolling_H24, K=3, signed_log1p
적용.

---

## 8. 산출물

`results/research_s1_phase2/`:
- `P2_test_multi_seed.csv` — 5 seed × test 지표
- `P2_rf_baseline_multi_seed.csv` — 동일 seed RF 비교
- `P2_summary.json` — 통합 요약 (GRU-D + GRU + RF + bootstrap CI)
- `P2_pr_auc_comparison.png` — 3-way 비교 bar chart
- `P2_train_curves.png` — train/valid 곡선

`src/research/s1_irregular_ts/`:
- `sequences_grud.py` — combined_raw 시작, NaN 보존, feature-level mask/Delta
- `gru_d.py` — Che et al. 2018 GRU-D 구현 (Input/Hidden decay)
- `run_s1_phase2.py` — 5-seed run + Phase 1, RF 비교
