# 시계열 모델링 — Phase 3: Transformer + Time2Vec

## K=3 짧은 시계열에서도 self-attention이 GRU 계열을 이기는가?

작성일: 2026-05-30
작성자: 이현지
관련 계획: `docs/다음 실험 설계안(연구지향 Stream 0).md` §Stream 1
선행 연구: `docs/research_s0_diagnostic_report.md`, `docs/research_sr_labeling_report.md`,
`docs/research_s1_phase1_report.md`, `docs/research_s1_phase2_report.md`
재현 코드: `src/research/s1_irregular_ts/`
결과 산출물: `results/research_s1_phase3/`

---

## 초록 (Abstract)

Phase 2 GRU-D는 "결측 처리 정교화"가 본 데이터 규모에서는 도움이 안 된다는
negative result를 남겼다 — 추가 파라미터 ~2,242개가 양성 641개에서 overfit
했다. Phase 3는 그와 직교하는 가설을 검증한다: **시계열 표현 자체를 self-
attention + 학습 가능한 시간 임베딩(Time2Vec)으로 다양화하면 어떤가**.
K=3이라는 짧은 시퀀스에서 self-attention의 효용성은 본질적으로 의문스럽지만,
Phase 1/2 결과가 이 비교를 자연스러운 다음 단계로 만든다.

모델은 Time2Vec (Kazemi et al. 2019, time_dim=8) + 1-layer Transformer
Encoder (model_dim=64, num_heads=4, GELU, pre-norm) + mean pooling.
Phase 1과 동일한 33-feature 시퀀스 + L3 rolling_H24 라벨링 + 동일 5-seed
프로토콜. Hyperparameter는 의도적으로 보수적(소형)으로 두어 Phase 1 GRU와의
fair comparison을 유지.

다섯 가지 결과가 Phase 3의 강한 우위를 보여준다.
**(1)** Transformer의 5-seed 평균 test PR-AUC는 **0.342 ± 0.023**으로
Phase 1 GRU의 0.304 ± 0.046 대비 +0.038, RF baseline 대비 +0.073.
**(2)** std는 0.023으로 Phase 1 GRU의 **절반** 수준 — *학습 안정성이 함께
개선되었다*는 가장 중요한 발견.
**(3)** Top-20 precision = **0.70**으로 다른 모든 모델(0.56~0.58)을 압도 —
"위험 검출" 운영 지표에서 명확한 우위.
**(4)** ROC-AUC = 0.889로 4-way 비교 중 1위 — ranking 자체도 가장 강력.
**(5)** Ensemble bootstrap point는 Phase 1 GRU(0.376)가 Transformer(0.363)
보다 약간 높지만 CI 폭이 더 좁아 (Transformer [0.247, 0.500] vs GRU
[0.257, 0.513]) 사실상 동등. 5-seed의 prediction 다양성이 GRU 쪽이 약간 더 커
ensemble의 이득이 큰 것이지, 단일 seed의 신뢰성은 Transformer가 압도적이다.

이로써 **Phase 3 Transformer가 본 stream의 새 best baseline**이 된다.
Phase 1 GRU가 가졌던 "+0.106 ensemble 이득" 보다 "+0.038 mean & −0.023 std"
조합이 운영 관점에서 더 가치 있다 — 평가가 불안정한 (valid 양성 39) 환경
에서 seed-variance 절반 감소는 hyperparameter 선택의 신뢰도를 끌어올린다.
Phase 4 (Neural CDE) 는 GPU 자원이 확보되면 시도, 그 전까지는 Phase 3
Transformer를 default로 두고 정규화·앙상블 다양화에 집중할 가치가 있다.

---

## 1. 배경: Phase 2가 남긴 방향

Phase 2 GRU-D의 negative result는 후속 phase 설계에 직접적인 지침을 줬다.

> 본 데이터 규모(641 train pos, 39 valid pos)에서는 모델 복잡도를 늘리는
> 방향보다 **시계열 표현 자체를 다양화**하거나 **정규화를 강화**하는 방향
> 이 더 유망하다.

Self-attention + Time2Vec는 그 첫 번째 방향(시계열 표현 다양화)의 자연스러운
다음 단계다. Phase 1 GRU가 hidden state를 순차적으로 갱신하는 반면, attention
은 **모든 timestep을 동시에 고려**하여 가중치를 학습한다. K=3이라는 짧은
시퀀스에서는 attention의 long-range 의존성 이점이 별로 없지만, **각 timestep
의 표현이 어떻게 결합되는지에 대한 추가 자유도**가 작용할 수 있다.

Time2Vec (Kazemi et al. 2019) 의 동기는 더 구체적이다.

- Positional encoding이 시계열 도메인에 부적합하다는 인식 (NLP의 word position
  vs 시계열의 절대/상대 시간).
- 학습 가능한 sin/cos + 선형 성분으로 시간을 임베딩 → 모델이 "1년 차이"의 의미
  를 데이터-driven으로 학습.
- K=3 같은 짧은 시퀀스에서도 의미 있는 시간 차원 신호 추가.

본 phase는 두 아이디어를 결합한 가장 단순한 형태를 평가한다.

---

## 2. 실험 설정

### 2.1 데이터

Phase 1과 동일:
- `processed_fixed_v1`의 imputed 데이터 (33 features = 27 재무 + 3 YoY + 6
  거시 - 매출액증가율 등 HIGH_MISSING 제외 후 = 실제로 27+6=33)
- `signed_log1p` 스케일링
- K=3 sequence, train 라벨 = L3 rolling_H24, valid/test 정답 = L1 fixed_N1
- (N, K, F) = train (29,986, 3, 33), valid (5,050, 3, 33), test (5,311, 3, 33)
- 양성: train 641 (L3) / valid 39 / test 56 (L1)

### 2.2 모델

```
Time2Vec(t):  τ_0(t) = ω_0 · t + φ_0    (선형 성분)
              τ_i(t) = sin(ω_i · t + φ_i)  i ∈ [1, time_dim-1]

Input concat: [X (B,K,33) ; Time2Vec(t) (B,K,8)]   → (B, K, 41)
Linear proj:  → (B, K, model_dim=64)
TransformerEncoder × 1: pre-norm, 4 heads, FFN dim=128, GELU, dropout=0.2
Pooling: mean over observed positions (mask-aware)
Head: Linear(64 → 32) → ReLU → Dropout → Linear(32 → 1)
```

학습 hyperparameters:
- Loss: focal (γ=2.0, α=0.25) — Phase 1, 2와 동일
- Optimizer: Adam, lr=**5e-4** (Phase 1의 1e-3보다 낮춤, Transformer 안정성 위해)
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=3) on valid PR-AUC
- Batch=256, epochs=30, patience=8

총 trainable parameter: ~22,000 (Phase 1 GRU ~17,000과 비슷하며 GRU-D
~19,000보다 약간 더 많음 — 그러나 추가가 W_dec_x/h 같은 task-specific 파라미터
가 아닌 **공통 attention/feedforward 가중치**라는 점에서 정규화 효과가 다르다).

### 2.3 평가

- 5 seed (42, 7, 13, 21, 100) × 1 split, 동일 fixed_N1 valid/test 정답.
- 5-seed ensemble bootstrap CI (n=2000).
- 4-way 비교: Phase 3 (Transformer+T2V), Phase 1 (GRU), Phase 2 (GRU-D), RF.

---

## 3. 결과

### 3.1 5-seed 측정 (Phase 3)

| seed | best_ep | valid PR | **test PR** | test ROC | P@20 | P@50 |
|------|---------|----------|--------------|----------|------|------|
| 42   | 14      | 0.129    | **0.366**    | 0.883    | 0.75 | 0.36 |
| 7    | 22      | 0.188    | 0.310        | 0.906    | 0.60 | 0.36 |
| 13   | 8       | 0.112    | 0.354        | 0.885    | 0.70 | 0.42 |
| 21   | 22      | 0.161    | 0.353        | 0.889    | 0.75 | 0.42 |
| 100  | 17      | 0.128    | 0.327        | 0.882    | 0.70 | 0.38 |

특기 사항:
- 모든 seed가 test PR-AUC 0.31~0.37 사이 (Phase 1 GRU는 0.265~0.382의 더 넓은
  범위, Phase 2 GRU-D는 0.187~0.353의 가장 넓은 범위).
- best_epoch이 8~22로 다양 — 학습이 빠르게 정점에 도달하는 seed도 있지만 그
  결과가 다른 seed에 비해 나쁘지 않다. early stop 패턴이 다양해도 결과는 안정.

### 3.2 4-way 비교 표

| 모델 | 5-seed mean ± std | Ensemble point | 95% CI | ROC-AUC | P@20 | P@50 |
|------|---------------------|-------------------|---------|---------|------|------|
| **Transformer+T2V (P3)** | **0.342 ± 0.023** | 0.363 | [0.247, 0.500] | **0.889** | **0.70** | **0.388** |
| GRU (P1) | 0.304 ± 0.046 | **0.376** | **[0.257, 0.513]** | 0.853 | 0.56 | 0.380 |
| GRU-D (P2) | 0.267 ± 0.068 | 0.315 | [0.208, 0.442] | 0.875 | 0.56 | 0.328 |
| RF baseline (S0) | 0.269 ± 0.015 | 0.270 | [0.166, 0.408] | 0.855 | 0.58 | 0.368 |

**Transformer가 5-seed mean / ROC-AUC / P@20 / P@50 4개 지표에서 1위**.
GRU는 ensemble bootstrap point에서만 1위. RF와 GRU-D는 모든 지표에서 후순위.

> 그림: `results/research_s1_phase3/P3_pr_auc_comparison.png`

### 3.3 안정성: std와 학습곡선

Phase별 seed-variance 비교:

| 모델 | 5-seed std | range (max − min) |
|------|------------|---------------------|
| Transformer+T2V (P3) | **0.023** | 0.056 |
| GRU (P1) | 0.046 | 0.118 |
| GRU-D (P2) | 0.068 | 0.166 |
| RF baseline | 0.015 | (5-seed) |

Transformer가 RF baseline 다음으로 가장 안정. Phase 1 GRU 대비 절반, Phase 2
GRU-D 대비 1/3 수준. **딥 모델 중에서는 가장 안정**.

`results/research_s1_phase3/P3_train_curves.png`에서:
- Train loss는 epoch 5~10 사이에 수렴.
- Valid PR-AUC가 ep 8~22 사이 정점 도달 → patience=8 early stop이 작동.
- best_epoch 분포가 8/14/17/22/22 — Phase 1 GRU(13~23)와 유사한 범위에서
  early stop 발생.

### 3.4 운영 지표 (Top-K precision/lift)

| 모델 | P@20 | P@50 | Lift@20 (= P@20 / base rate 0.0105) |
|------|------|------|----------------------------------------|
| Transformer+T2V | **0.700** | **0.388** | **66.6×** |
| GRU (P1) | 0.560 | 0.380 | 53.3× |
| GRU-D (P2) | 0.560 | 0.328 | 53.3× |
| RF baseline | 0.580 | 0.368 | 55.2× |

가장 위험한 20개 기업을 골랐을 때 Transformer는 14개를 정답으로 맞춤 (다른
모델은 11~12개). 운영 관점에서 가장 유의미한 개선.

---

## 4. 해석

### 4.1 왜 Transformer가 K=3에서도 GRU를 이겼는가

K=3은 attention의 long-range 의존성 이점을 별로 활용할 수 없는 짧은 시퀀스다.
그럼에도 Transformer가 우위인 이유에 대한 가설:

1. **Time2Vec의 효과**. Phase 1 GRU는 timestep 순서를 RNN의 순환 구조로
   암묵적으로 학습한다. Time2Vec는 명시적 시간 임베딩을 제공해 "1년 차이 ≠ 2년
   차이"를 데이터로부터 학습하게 한다. 특히 결측 timestep이 padding으로 들어
   갔을 때 (mean_obs_steps = 2.50/2.82/2.83), 시간 임베딩이 어느 timestep이
   실제 관측인지 함께 사용된다.
2. **Pre-norm + GELU 안정성**. Transformer Encoder의 pre-norm 구조와 GELU
   activation은 학습 초기 gradient flow를 안정화한다. Phase 1 GRU의 tanh 기반
   gating은 작은 양성 신호에서 saturation에 빠지기 쉽다.
3. **Mean pooling의 정규화 효과**. Transformer는 K timestep의 표현을 평균내며
   결합한다. RNN은 마지막 timestep에 과도하게 의존할 수 있다. 평균은 그
   자체로 정규화 역할.
4. **Attention 가중치 학습**. K=3에서도 어떤 timestep이 분류에 더 중요한지를
   학습한다. 예: 가장 최근 timestep이 신호를 가장 강하게 가질 때 self-attention
   이 그것에 집중하도록.

### 4.2 GRU vs Transformer: ensemble과 single seed의 비대칭

흥미로운 관찰: **Phase 1 GRU의 ensemble (0.376)이 Phase 3 Transformer ensemble
(0.363)보다 약간 높다**. 그러나 5-seed mean은 반대로 Transformer가 +0.038
높다.

이는 GRU의 seed-variance가 더 크다는 사실(0.046 vs 0.023)과 직접 연결된다.
- GRU 5개 모델의 prediction 확률이 더 다양하다 → 평균으로 ensemble하면 큰 이득
  (mean 0.304 → ensemble 0.376, +0.072).
- Transformer 5개 모델의 prediction은 일관적이다 → ensemble 이득이 작다
  (mean 0.342 → ensemble 0.363, +0.021).

운영 관점에서 어느 쪽이 나은가?
- ensemble 효과를 활용할 수 있다면 GRU가 약간 더 좋다.
- 단일 모델만 배포해야 한다면 Transformer가 명확히 우위.
- **단일 seed로 hyperparameter를 선택해야 하는 환경**에서는 Transformer가
  훨씬 신뢰성 있다 (range 0.056 vs 0.118).

### 4.3 ROC-AUC와 PR-AUC가 모두 1위

Phase 2 GRU-D는 ROC-AUC가 높지만 PR-AUC가 낮은 비대칭이 있었다. Phase 3
Transformer는 **양쪽 모두 1위**. 이는 양성과 음성의 분리뿐만 아니라 양성 영역
내의 정밀한 ranking까지 잘한다는 의미다. Top-20 precision 0.70이 그 직접
증거.

### 4.4 본 stream의 새 baseline

Phase 1 GRU가 가졌던 "ensemble +0.106" 자리는 Phase 3 Transformer의 "mean
+0.038, std -0.023, P@20 +0.14" 조합으로 교체된다. **단일 모델 신뢰성 + top-K
운영 지표**가 우선되는 본 task에서는 Phase 3가 명백한 best.

---

## 5. 한계

- **단일 split** (walk-forward CV 미수행). Phase 1, 2와 동일한 약점. 가장
  어려운 valid year인 2023 사용.
- **K=3 고정**. K=4, 5에서 Transformer의 attention이 어떻게 변하는지 미검증.
  K가 길수록 Transformer 우위가 더 커질 가능성.
- **1-layer Transformer만**. 2-layer/3-layer ablation 미수행. K=3에서는 1
  layer로 충분할 가능성이 높지만 확인 필요.
- **Hyperparameter sweep 없음**: model_dim 64, heads 4, time_dim 8, lr 5e-4
  의 단일 설정만 평가. 5-seed std가 작으므로 sweep 비용이 낮은 편이라 다음
  단계에서 시도 가치.
- **Attention 가중치 해석 미수행**. K=3에서 어느 timestep이 평균적으로 가장
  중요한 가중치를 받는지 분석하면 모델 행동을 더 정확히 이해 가능.

---

## 6. 다음 단계 권장

| 항목 | 권장 |
|------|------|
| **Phase 4 Neural CDE** | **GPU 확보 시 권장** — UNIST DAL 시그니처. 본 stream의 학술적 차별성. K가 길어질수록 우위 가능. |
| **Walk-forward CV로 Phase 3 재검증** | **권장** — 새 best baseline의 robustness 확인. CPU에서 ~3시간 예상. |
| **K sensitivity (K=2, 4, 5)** | **권장** — K=3 외의 길이에서 우위가 유지되는지. Phase 1, 2, 3 모두 테스트. |
| **Phase 3 + ensemble 다양화** | **선택** — Phase 1 GRU와 Phase 3 Transformer prediction을 결합 (weighted average, stacking) → 두 모델의 다른 강점을 활용. |
| **Phase 3 Attention 가중치 분석** | **선택** — paper-style 보고를 위해서는 해석 가치. |

---

## 7. 재현성

```powershell
# 단일 seed 빠른 점검
.venv\Scripts\python.exe -m src.research.s1_irregular_ts.run_s1_phase3 --quick

# 5-seed full
.venv\Scripts\python.exe -m src.research.s1_irregular_ts.run_s1_phase3
```

Default hyperparameter: K=3, labeling=L3_rolling_H24, model_dim=64, num_heads=4,
num_layers=1, dropout=0.2, time_dim=8, lr=5e-4, batch=256, epochs=30, patience=8.

추가 의존성 없음 (torch 2.12 CPU + 기존).

---

## 8. 산출물

`results/research_s1_phase3/`:
- `P3_test_multi_seed.csv` — 5 seed × test 지표
- `P3_rf_baseline_multi_seed.csv` — 동일 seed RF 비교
- `P3_summary.json` — 통합 요약 (Transformer + GRU + GRU-D + RF + bootstrap CI)
- `P3_pr_auc_comparison.png` — 4-way bar chart
- `P3_train_curves.png` — train loss + valid PR-AUC 곡선

`src/research/s1_irregular_ts/`:
- `transformer_t2v.py` — Time2Vec + Transformer Encoder 분류기
- `run_s1_phase3.py` — 5-seed run + 4-way 비교 + bootstrap CI
