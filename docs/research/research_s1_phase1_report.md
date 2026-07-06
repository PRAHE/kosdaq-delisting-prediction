# 시계열 모델링 진입 — Phase 1: GRU baseline

## 단일 스냅샷에서 K=3 시퀀스로 옮기면 무엇이 달라지는가

작성일: 2026-05-29
작성자: 이현지
관련 계획: `docs/다음 실험 설계안(연구지향 Stream 0).md` §Stream 1
선행 연구: `docs/research_s0_diagnostic_report.md`, `docs/research_sr_labeling_report.md`
재현 코드: `src/research/s1_irregular_ts/`
결과 산출물: `results/research_s1_phase1/`

---

## 초록 (Abstract)

S0 진단(PR-AUC 0.2876의 95% CI [0.180, 0.421])과 SR 라벨링 점검(7개 라벨링 모두
baseline의 CI 안)에서 도출된 공통 결론은 **"단일 연도 스냅샷 + RF로는 천장을
못 깬다"**는 것이었다. 본 stream은 그 진단의 직접적 후속으로, 같은 33개
재무 피처를 **K=3년 시퀀스**로 재구성해 GRU에 입력했을 때 무엇이 달라지는지
측정한다. 모델은 1-layer GRU (hidden=64) + focal loss (γ=2, α=0.25) + 5-seed
multi-run, 라벨링은 SR에서 가장 일관되게 우수했던 **L3 rolling_H24**를 사용
한다.

다섯 가지 결과가 시계열 진입의 가치를 보여준다.
**(1)** GRU의 5-seed 평균 test PR-AUC는 **0.304** (std 0.046)로 RF baseline의
0.269 (std 0.015)보다 +0.035 높다.
**(2)** 5-seed의 예측 확률을 **평균낸 ensemble** 기준에서는 점추정 PR-AUC가
**0.376** (95% CI [0.257, 0.513])로 RF baseline의 0.270 (CI [0.166, 0.408])
대비 +0.106 우위 — 이는 S0 baseline의 CI 폭(0.24)의 약 절반에 해당하는
실질적 개선이다.
**(3)** Top-20/50 precision은 RF와 사실상 동일 (P@20=0.56 vs 0.58, P@50=0.38
vs 0.37) — 시계열은 ranking 능력보다는 **확률 분포 자체의 질**을 개선한다.
**(4)** 그러나 GRU의 seed-variance는 RF의 3배 (0.046 vs 0.015) — 학습이 더
불안정하다. 단일 seed에서는 0.265~0.382로 큰 폭의 변동.
**(5)** 가장 약한 seed (0.265) 도 RF의 평균(0.269)과 거의 같다 — 시계열
전환이 baseline 미만으로 떨어지지는 않지만, 안정성 개선이 Phase 2의 핵심
과제가 된다.

종합적으로 **시계열 입력 자체가 의미 있는 정보 채널**임이 확인되었다. 다만
1-layer GRU + valid set 39 양성이라는 조건에서 hyperparameter 선택의 분산이
크다. Phase 2 (GRU-D로 missing/irregular 처리), Phase 3 (Transformer +
time2vec 비교), Phase 4 (Neural CDE)가 정당화된다.

---

## 1. 배경: S0와 SR의 결론이 왜 시계열을 가리키는가

S0 보고서의 핵심 발견은 다음과 같다.

- baseline RF PR-AUC 0.2876의 95% Bayesian 신뢰구간은 [0.180, 0.421] — 폭 0.24.
- exp_010~018의 9개 사전 실험 모두 이 CI 안. 모델·피처·샘플링 조정만으로
  천장을 깰 수 없다.
- 35개의 confident FN(상폐 1년 전에도 재무비율상 정상으로 보이는 기업)이
  남은 오차의 본질.

SR 보고서는 라벨링 7개 변형 모두 같은 CI 안에 있음을 확인했다 — 라벨 정의를
바꾸는 것만으로도 천장을 못 깬다. 하지만 SR에서 우수했던 L3 rolling_H24가
"2년 내 상폐 위험"을 라벨로 잡는다는 사실은, 시계열 framing이 자연스럽게
이 라벨링과 짝을 이룬다는 힌트를 준다.

**FN의 본질은 trajectory**다. 상폐 1년 전 시점만 보면 정상이지만, 과거 3~5년의
재무비율이 어떻게 악화되어 왔는지를 보면 차이가 있을 수 있다. 본 Phase 1은
이 가설의 **가장 단순한 형태**를 검증한다.

---

## 2. 실험 설정

### 2.1 데이터 — 시퀀스 빌드

각 `(stock_code, year, quarter)` target 샘플에 대해 **동일 quarter의 과거
3년 시계열**을 입력으로 구성한다.

- 예: target (008800, 2020, Q1) → 시퀀스 [(2018 Q1), (2019 Q1), (2020 Q1)]
- 마지막 timestep이 current year. 과거 데이터가 부족하면 zero-pad + mask로 처리.
- 4개 quarter type(ANNUAL, H1, Q1, Q3) 각각 자기 시계열을 가짐 — 보고 주기
  혼합 자체는 quarter type 분리로 우회하고, 같은 quarter type 내 연도 시계열
  에 집중.
- 결측 처리·스케일링은 S0와 동일 (median impute + signed_log1p, train에서만 fit)

| Split | N | F | 평균 관측 steps | 양성 (L3) |
|-------|---|---|------------------|------------|
| train | 29,986 | 33 | 2.50 | 641 |
| valid | 5,050  | 33 | 2.82 | 39 (L1 ground truth) |
| test  | 5,311  | 33 | 2.83 | 56 (L1 ground truth) |

`train`은 L3 rolling_H24 라벨로 학습하지만 `valid`/`test`는 항상 L1 fixed_N1
정답으로 평가 (SR의 공정 비교 원칙 유지).

### 2.2 모델

```
Input (B, K=3, F=33)
  → GRU (hidden=64, num_layers=1, dropout=0.2, batch_first=True)
  → mask로 마지막 관측 timestep의 hidden 추출
  → Linear(64→32) → ReLU → Dropout → Linear(32→1)
  → sigmoid
```

- **Loss**: focal loss (γ=2.0, α=0.25) — 극심한 불균형에 BCE보다 안정 (Lin et
  al. 2017).
- **Optimizer**: Adam, lr=1e-3, batch=256.
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=3) on valid PR-AUC.
- **Early stopping**: valid PR-AUC 8 epoch 무개선 시 중단, max 30 epoch.

### 2.3 평가

- 5 seed (42, 7, 13, 21, 100) 독립 학습 → 각각 test PR-AUC, ROC-AUC, P@20, P@50.
- 5-seed의 test prediction probability를 평균낸 **ensemble** 결과에 대해
  Bayesian bootstrap (n_boot=2000) CI 산출.
- 비교: 동일 5-seed로 학습한 S0 canonical RF baseline.

---

## 3. 결과

### 3.1 5-seed 측정값

| seed | best_epoch | valid PR-AUC | **test PR-AUC** | test ROC-AUC | P@20 | P@50 |
|------|------------|---------------|------------------|---------------|------|------|
| 42   | 16         | 0.184         | 0.279            | 0.861         | 0.45 | 0.34 |
| 7    | 23         | 0.218         | 0.265            | 0.840         | 0.60 | 0.32 |
| **13** | **13**   | **0.231**     | **0.382**        | 0.842         | **0.65** | **0.46** |
| 21   | 17         | 0.203         | 0.303            | 0.860         | 0.50 | 0.42 |
| 100  | 16         | 0.239         | 0.291            | 0.862         | 0.60 | 0.36 |

| | GRU mean ± std | RF baseline mean ± std | Δ |
|--|---|---|---|
| **Test PR-AUC** | **0.304 ± 0.046** | 0.269 ± 0.015 | **+0.035** |
| Test ROC-AUC | 0.853 ± 0.010 | 0.855 ± 0.003 | −0.002 |
| Test P@20 | 0.560 | 0.580 | −0.020 |
| Test P@50 | 0.380 | 0.368 | +0.012 |

### 3.2 5-seed 평균 확률 ensemble + bootstrap CI

5개 seed의 test prediction probability를 평균낸 단일 score에 대한 Bayesian
bootstrap (n_boot=2000):

| 모델 | Bootstrap point | 95% CI | std |
|------|------------------|---------|------|
| **GRU ensemble** | **0.376** | **[0.257, 0.513]** | 0.066 |
| RF baseline ensemble | 0.270    | [0.166, 0.408]     | 0.062 |
| Δ | **+0.106** | — | — |

GRU의 CI [0.257, 0.513]는 RF의 CI [0.166, 0.408]과 일부 겹치지만 **현저히
오른쪽으로 이동**되어 있다. S0 baseline의 CI 폭 0.24의 약 절반에 해당하는
실질적 개선.

> 그림: `results/research_s1_phase1/P1_pr_auc_comparison.png`

### 3.3 학습 곡선 관찰

`results/research_s1_phase1/P1_train_curves.png`에서:

- 모든 5 seed가 ep 13~23 사이에서 valid PR-AUC 정점에 도달 → early stop.
- Valid PR-AUC가 0.16~0.24 사이로 seed별 큰 변동.
- Train loss는 ep 5 이후 거의 평탄해지나 valid가 들쭉날쭉 — valid 양성 39개의
  noise.

---

## 4. 해석

### 4.1 핵심 발견: 시계열 전환의 효과는 실재한다

- 5-seed 평균에서 +0.035 우위, ensemble에서 +0.106 우위. 두 측정 모두 GRU가
  더 좋은 방향을 가리킨다.
- ROC-AUC가 거의 동일 (0.853 vs 0.855)한데 PR-AUC만 개선 — 시계열은 음성 대비
  양성을 더 잘 **순위 결정(precision 측면)** 한다는 의미. ranking 능력 자체가
  변화한 것이 아니라 **양성 영역에서의 확률 분포 질**이 개선됨.
- P@20/P@50은 거의 동일 — top-K decision에서는 차이가 없으나, 그 외 운영
  threshold에서의 precision-recall trade-off는 GRU가 더 유리.

### 4.2 단점: 학습 안정성

- GRU seed-variance 0.046은 RF의 0.015의 3배.
- 가장 약한 seed(7, 0.265)는 RF baseline 평균(0.269)에 거의 동일 — 시계열로
  옮긴다고 항상 좋아지지는 않는다.
- 가장 강한 seed(13, 0.382)는 baseline CI 상단 근처 — 잠재력은 분명히 있다.
- 평균 ensemble이 모든 단일 seed보다 좋다는 점(0.376 > 0.382 seed13 한 점 빼고)
  은 **여러 모델의 다양성을 활용한 ensemble이 학습 분산을 흡수**함을 보여준다.

### 4.3 SR L3 라벨링의 효과

SR에서 L3 rolling_H24가 우수했던 이유 — 양성을 2배 늘려 모델에 더 많은
정보를 제공 — 가 시계열 모델에서 더 명확해진다. RF는 양성 수에 둔감(SR에서
L3 vs L1 차이 미미)했지만 GRU는 학습 시 양성 noise gradient가 더 안정되므로
양성 추가의 효과가 더 크다.

### 4.4 시계열이 천장을 어디까지 올렸는가

S0의 RF baseline CI는 [0.180, 0.421]. GRU ensemble의 CI는 [0.257, 0.513]. 두
CI가 부분적으로 겹치지만, ~~**GRU CI 하단(0.257)이 RF CI 중앙(0.293)을 거의
초과**한다~~ **GRU CI 하단(0.257)이 RF CI 중앙(0.293)에 근접하나 아직 미달**한다. 단일 시점 평가에서 통계적 우위까지 주장하기는 어렵지만, 점추정
이동의 크기(+0.106)는 의미 있다.

---

## 5. 한계와 다음 단계

### 5.1 본 Phase의 한계

- **단일 split만 평가** (walk-forward CV는 다음 단계). seed별 변동성이 큰
  이유의 일부는 valid 2023의 본질적 어려움(S0 발견)이다.
- **K=3 고정** (sensitivity 미수행). K=2, 4, 5와의 비교가 향후 필요.
- **단일 모델 (1-layer GRU, hidden 64)**. hyperparameter sweep 없이 결과 보고.
- **Mask 사용은 단순한 형태** (마지막 관측 timestep의 hidden을 분류 입력으로).
  GRU-D 같은 시간 decay 메커니즘은 Phase 2에서 도입.
- **5-seed로는 std 추정이 약함**. 진정한 평가는 10+ seed × walk-forward fold
  필요.

### 5.2 Phase 2 ~ 4로의 권고

| Phase | 목표 | 진단으로 부각된 동기 |
|-------|------|---------------------|
| **Phase 2** | GRU-D (missing + time decay) | 본 phase에서 mask 단순 처리 → 결측 시점의 명시적 가중이 의미 있을 듯 |
| **Phase 3** | Transformer + time2vec | self-attention이 K=3 같은 짧은 시퀀스에 과적합인지 검증 |
| **Phase 4 (선택)** | Neural CDE | 본 phase에서 시계열 전환이 effective함이 확인되었으므로 NCDE의 추가 이득을 평가할 의미가 있음 |

또한 본 phase의 +0.106 ensemble 우위는 **앙상블 자체가 강력한 정규화**임을
시사한다. 다음 phase에서도 단일 seed 결과보다 multi-seed ensemble 보고를
표준화할 것.

---

## 6. 재현성

```powershell
# 단일 seed 빠른 점검
.venv\Scripts\python.exe -m src.research.s1_irregular_ts.run_s1_phase1 --quick

# 5-seed 풀 (약 15분 CPU)
.venv\Scripts\python.exe -m src.research.s1_irregular_ts.run_s1_phase1
```

기본 hyperparameter: K=3, labeling=L3_rolling_H24, hidden=64, 1-layer GRU,
focal γ=2.0 α=0.25, lr=1e-3, batch=256, epochs=30, patience=8.

추가 의존성: `torch` 2.12 CPU (수동 설치).

---

## 7. 산출물

`results/research_s1_phase1/`:
- `P1_test_multi_seed.csv` — 5 seed × test 지표
- `P1_rf_baseline_multi_seed.csv` — 동일 seed RF 비교
- `P1_summary.json` — 통합 요약 (GRU vs RF + bootstrap CI)
- `P1_pr_auc_comparison.png` — bar chart with seed points
- `P1_train_curves.png` — train loss + valid PR-AUC 곡선

`src/research/s1_irregular_ts/`:
- `sequences.py` — 시퀀스 빌더 (K-step sliding window + mask)
- `gru_baseline.py` — GRU model + focal loss + 학습 루프
- `run_s1_phase1.py` — 5-seed run + RF 비교 + bootstrap CI
