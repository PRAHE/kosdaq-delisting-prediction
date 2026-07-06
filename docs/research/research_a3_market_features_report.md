# 외부 데이터 (A3) — 주가·거래량 시장 피처 추가

## 시장 데이터 6개로 천장을 깰 수 있는가?

작성일: 2026-05-30
작성자: 이현지
관련 계획: `docs/다음 실험 설계안(모델 개선).md` §A3
선행 연구: S0 진단 / SR 라벨링 / S1 Phase 1~3 / S1 walk-forward
재현 코드: `src/research/a3_market/`, `src/research/s1_irregular_ts/`
결과 산출물: `results/research_s1_walkforward_market/`,
`preprocess/data/market/market_features.csv`

---

## 초록 (Abstract)

S1 walk-forward 재검증은 Phase 3 Transformer가 4/4 fold에서 GRU와 RF baseline
을 능가함을 확립했으나, 전체 walk-forward 평균 PR-AUC는 0.289로 S0 baseline
의 95% Bayesian CI [0.180, 0.421] 여전히 안쪽이었다. 본 보고서는 외부 데이터
도입의 첫 단계로 **A3 주가·거래량 시장 피처 6개**를 추가한 결과를 보고한다.

데이터: FinanceDataReader로 fixed_v1 패널의 1,945 종목 중 **1,516종목의
일별 OHLCV** (2014~2025) 수집. 분기 말 시점 기준 6개 피처 산출 —
`price_log_close`, `price_ret_12m`, `price_volatility_60d`,
`price_drawdown_max_12m`, `volume_log_mean_60d`, `volume_change_yoy`.
전체 (stock, year, quarter) 45,854 키에 대해 93~100% 관측률 확보.

모델: 33 features (재무 27 + YoY 3) + 거시 6 → **39 features**.
나머지는 S1 walk-forward와 완전 동일: GRU + Transformer+T2V, K=3, L3
rolling_H24 라벨, 4 fold × 3 seed = 12 측정.

다섯 가지 결과가 **"의미 있지만 게임체인저 아님"** 이라는 결론을 만든다.
**(1)** Transformer walk-forward 평균 test PR-AUC는 0.289 → **0.297** (+0.008,
모든 fold +0.005~0.012로 일관). 작지만 통계적 noise보다는 크다.
**(2)** ROC-AUC는 명확히 개선 — Transformer 0.887 → **0.920** (+0.033),
GRU 0.864 → **0.918** (+0.054). **시장 피처가 ranking에는 강하게 기여**.
**(3)** 그러나 Top-20 precision은 **하락** (Transformer 0.575 → 0.500, −0.075).
**시장 피처는 전체 ranking을 정렬하지만 양성 영역 정밀도는 오히려 약화**.
**(4)** GRU가 Transformer보다 더 큰 폭으로 개선 (fold 1 GRU +0.050 vs
Transformer +0.007) — 데이터 적은 환경에서 시장 피처의 한계효용이 더 큼.
**(5)** Valid PR-AUC가 크게 올라감 (Transformer 0.255 → 0.308, +0.054)에 비해
test 개선은 +0.008에 그침 — **mild overfitting** 패턴.

종합: 시장 피처는 ROC-AUC와 fold 안정성에 도움되지만 **PR-AUC 천장(0.30
부근)을 못 깸**. 본 실험은 **A3 단독으로는 부족하며, 더 직접적 상폐 선행
신호(A1 감사의견·A2 관리종목)와의 결합이 필요**함을 보여준다.

---

## 1. 배경: 시장 데이터가 왜 도움될 것으로 기대됐는가

S1 walk-forward 보고서까지 우리는 다음 결론에 도달했다.

- 단일 스냅샷 RF (0.270 ± 0.015) vs 시계열 Transformer (walk-forward 0.289 ±
  0.056). 평균 +0.019 우위지만 baseline CI 안.
- 천장의 본질은 **상폐 1년 전에도 재무비율상 정상으로 보이는 35개 confident FN**
  (S0-A6의 발견).

이론적 근거:
- Campbell, Hilscher & Szilagyi (2008, JF): 시장 기반 변수(시가총액, 주가
  수익률, 변동성)가 회계 기반 변수보다 부실 예측력이 높음
- Shumway (2001, JB): 시장 변수 결합 hazard model이 Altman Z-score 능가
- 재무비율은 분기/연간 시점에 갱신되지만 **시장 가격은 매일 정보를 반영** —
  분포 이동에 robust할 가능성

본 phase는 이 가설을 직접 검증한다.

---

## 2. 데이터 수집 및 피처 산출

### 2.1 OHLCV 수집

- 라이브러리: FinanceDataReader 0.9.2
- 대상: fixed_v1 패널 train/valid/test에 등장하는 모든 unique stock_code (1,945개)
- 기간: 2014-01-01 ~ 2025-12-31 (lag/window 계산용으로 1년 추가 buffer)
- 수집 성공: **1,516종목 (78%)**, 실패 0 — 비상장 종목·구 코드는 skip
- 캐싱: `data/market_ohlcv/{stock_code}.csv` (1,516개 파일)
- Wall time: ~9분 30초 (6 workers 병렬)

### 2.2 6개 분기별 피처 (45,854 행)

분기 시점 매핑: Q1=3월 마지막 거래일, H1=6월, Q3=9월, ANNUAL=12월.

| 피처 | 정의 | 관측률 |
|------|------|--------|
| `price_log_close` | 분기 말 종가 로그 | 100.0% |
| `price_ret_12m` | (분기 말 종가) / (1년 전 종가) − 1 | 94.9% |
| `price_volatility_60d` | 직전 60거래일 일일 log-return std × √252 | 99.8% |
| `price_drawdown_max_12m` | (직전 252거래일 최고가 − 현재 종가) / 최고가 | 98.9% |
| `volume_log_mean_60d` | 직전 60거래일 평균 거래량 로그 | 99.9% |
| `volume_change_yoy` | 직전 60일 평균 거래량 / 1년 전 동일 기간 평균 − 1 | 92.7% |

이론적 매핑:
- **price_ret_12m, price_drawdown_max_12m**: Campbell+2008의 "minus past
  excess return" 핵심 변수
- **price_volatility_60d**: Bharath & Shumway 2008의 Merton DD 모형 핵심 입력
- **volume_log_mean_60d, volume_change_yoy**: 시장 유동성 지표 (자본조달 능력
  악화의 선행 신호)
- **price_log_close**: 작은 절대 가격 = 동전주 효과 (penny stock anomaly)

### 2.3 패널 결합

기존 fixed_N1 패널 (33 features)과 left-join → **39 features**.
fixed_v1에 있지만 시장 데이터에 없는 종목은 6개 시장 피처 NaN → 모델 내부
`SimpleImputer(median)`로 처리.

---

## 3. 실험 프로토콜

- S1 walk-forward와 완전 동일: 4 fold × 3 seed × 2 모델 = 24 학습
- Hyperparameter: GRU hidden=64, Transformer model_dim=64/heads=4/layers=1
- Loss: focal (γ=2.0, α=0.25), Adam, batch=256, epochs=25, patience=8
- 정답 라벨: 항상 L1 fixed_N1 (valid/test 통일)

---

## 4. 결과

### 4.1 모델 × fold × market 4-way 비교

**Transformer+T2V test PR-AUC (3-seed mean)**

| Fold | Valid year | Without market | With market | Δ |
|------|------------|----------------|-------------|-----|
| 1 | 2020 | 0.222 ± 0.040 | **0.229 ± 0.026** | +0.007 |
| 2 | 2021 | 0.277 ± 0.019 | **0.282 ± 0.052** | +0.005 |
| 3 | 2022 | 0.313 ± 0.045 | **0.325 ± 0.006** | +0.012 |
| 4 | 2023 | 0.343 ± 0.029 | **0.352 ± 0.021** | +0.009 |
| **평균** | | **0.289** | **0.297** | **+0.008** |

**4/4 fold 모두에서 시장 피처 추가가 작지만 일관된 개선** — 모든 fold +0.005
이상.

**GRU test PR-AUC**

| Fold | Without | With | Δ |
|------|---------|------|-----|
| 1 | 0.092 | **0.143** | **+0.050** |
| 2 | 0.190 | 0.204 | +0.014 |
| 3 | 0.259 | 0.267 | +0.008 |
| 4 | 0.309 | 0.304 | −0.005 |
| **평균** | **0.213** | **0.229** | **+0.016** |

GRU는 시장 피처로 더 큰 폭 개선 (특히 train 데이터 적은 fold 1 +0.050).
fold 4에서만 미세하게 하락.

### 4.2 전체 walk-forward 평균 (12 측정)

| 모델 | Variant | Test PR-AUC | Std | ROC-AUC | P@20 | P@50 |
|------|---------|-------------|-----|---------|------|------|
| GRU | 33 features | 0.213 | 0.094 | 0.864 | 0.371 | 0.285 |
| GRU | **39 (+market)** | **0.229** | **0.071** | **0.918** | 0.350 | 0.293 |
| Δ | | **+0.016** | **−0.023** | **+0.054** | −0.021 | +0.008 |
| Transformer | 33 features | 0.289 | 0.056 | 0.887 | **0.575** | 0.338 |
| Transformer | **39 (+market)** | **0.297** | 0.055 | **0.920** | 0.500 | **0.367** |
| Δ | | +0.008 | ≈ | **+0.033** | **−0.075** | +0.029 |

### 4.3 ROC-AUC가 크게 개선되는데 P@20은 왜 떨어지나?

가장 중요한 관찰. 시장 피처는:
- **전체 양/음성 ranking** (ROC-AUC)에 강하게 기여 — Transformer +0.033, GRU +0.054
- **양성 영역 top-20 정밀도**에는 오히려 부정적 — Transformer P@20 −0.075

해석 가설:
1. **시장 변동성·거래량 변화는 정상 기업에도 흔함**. 음성 중 "변동성 큰
   정상 기업"이 위로 올라와 top-20을 차지하여 진짜 양성을 밀어냄.
2. **양성 기업의 시장 신호는 이미 늦은 시점**. 상폐 1년 전 시점에는 시장 가격
   이 충분히 떨어져 있을 수 있지만, "큰 폭 하락" 자체는 회생 가능 정상 기업
   도 가짐.
3. **재무비율 기반 ranking이 양성 영역 분리에 더 정확**했으나, 시장 피처가
   추가 차원으로 들어가면서 모델의 양성 영역 결정 경계가 흐려짐.

이는 시장 피처의 **본질적 한계**다 — Campbell+2008의 결과를 정확히 재현하지
는 못한다 (그들의 데이터는 SP500 기반, 우리는 코스닥 위주).

### 4.4 Valid-Test 격차의 증가 = mild overfitting

| 모델 | Valid PR-AUC (mean) | Test PR-AUC (mean) | gap |
|------|----------------------|---------------------|-----|
| Transformer (33 features) | 0.255 | 0.289 | −0.034 (test가 valid보다 좋음) |
| Transformer (39 features) | **0.308** | **0.297** | **+0.011** (overfitting 시작) |

시장 피처 추가로 train fit이 개선되어 valid가 0.05 올라가지만 test는 +0.008만
오름. 즉 **시장 피처의 ~85%는 valid에서만 활용되고 test로 전이되지 않음**.

> 그림: `results/research_s1_walkforward_market/WF_comparison.png`

---

## 5. 정직한 해석

### 5.1 시장 피처는 "있으면 약간 좋지만 천장은 못 깸"

- 최대 Transformer PR-AUC: 0.289 → 0.297 (+0.008, ~3%)
- 모든 fold +0.005 이상으로 noise보다는 큼
- 그러나 S0 baseline CI [0.180, 0.421] 안에 머무름
- ROC-AUC 개선(+0.033)은 명확하지만 PR-AUC가 안 따라옴

### 5.2 우리 데이터에서 시장 피처가 약한 이유

- **코스닥 위주 (소형주)** → 시장 가격이 펀더멘털을 덜 정확히 반영. 동전주
  관리 종목의 가격은 노이즈가 크다.
- **상폐 직전이라도 거래정지 빈번**. 마지막 거래일 가격이 펀더멘털 신호로
  부정확할 수 있음.
- **이미 알고 있는 부실 기업**이 시장 변동성으로 식별되는 것은 큰 추가 정보
  아닐 가능성 — 재무비율이 이미 그것을 표현했음.

### 5.3 학술적 가치

본 phase의 결과는 **"Campbell+2008의 결과는 한국 코스닥에서 부분적으로만
재현된다"**는 학술적으로 의미 있는 negative-ish 발견이다. 시장 변수의 PR-AUC
공헌은 미국 S&P500 데이터에서보다 작다.

---

## 6. 다음 단계 권장 (갱신)

본 phase 결과로 후속 권장:

| 옵션 | 기대 효과 | 비용 | 권장도 |
|------|-----------|------|--------|
| **A1 감사의견** | ★★★★★ (KRX 상폐 사유 직결) | DART 파싱 노동 | **강력 권장** |
| **A2 관리종목 이력** | ★★★★★ (80%+ 상폐 기업 경험) | KIND 크롤링 / 라벨 누수 검토 | **강력 권장** |
| A3 시장 피처 시계열로 확장 | +0.01 추가 가능 | 작음 (월별 K=12) | 선택 |
| 시장 + 감사 + 관리종목 결합 | **+0.10~0.15 기대** | 데이터 결합 노동 | **궁극적 목표** |
| 현재까지로 portfolio 완성 | 0 | 시간만 | 일정 부족 시 선택 |

본 보고서까지의 결론:
- **시장 피처는 ranking 보강용 보조 신호 (ROC-AUC ↑)**
- **PR-AUC 천장 0.30 부근을 깨려면 A1/A2 같은 직접 신호 필요**
- **현재 best baseline**: Transformer+T2V (with market) walk-forward 평균 0.297

---

## 7. 한계

- **1,516종목 (전체의 78%)만 시장 데이터 보유**. 나머지 22%는 시장 피처 NaN
  → median imputation. 이 22%가 특히 상폐 위험이 큰 종목이면 시장 피처의 신호
  를 흐림.
- **분기 말 단일 시점 스냅샷만 사용**. 시장 데이터의 진짜 가치는 일별/주별
  세분도에 있을 수 있는데, 본 phase는 분기 단위로 집계했다.
- **3 seed만 사용** (S1 walk-forward와 동일). std 추정이 약함.
- **거시변수와의 교호작용 미수행**. 시장 변수와 거시변수(VIX 등)는 중복 가능.

---

## 8. 재현성

```powershell
# 1) OHLCV 수집 (한 번만, ~10분)
.venv\Scripts\python.exe -m src.research.a3_market.fetch_ohlcv --workers 6

# 2) 분기별 피처 산출 (~4분)
.venv\Scripts\python.exe -m src.research.a3_market.market_features

# 3) Walk-forward with market features (~80분)
.venv\Scripts\python.exe -m src.research.s1_irregular_ts.run_s1_walkforward \
    --seeds 42,7,13 --epochs 25 --with-market
```

---

## 9. 산출물

`src/research/a3_market/`:
- `fetch_ohlcv.py` — FinanceDataReader 기반 일별 OHLCV 수집 + 캐시
- `market_features.py` — 6개 분기별 시장 피처 산출

`data/market_ohlcv/` — 1,516종목 일별 OHLCV CSV
`preprocess/data/market/market_features.csv` — 45,854 행 × 9 컬럼 (3 메타 + 6 피처)

`results/research_s1_walkforward_market/`:
- `WF_grid.csv` — 24 측정 결과
- `WF_fold_summary.csv` — fold별 mean ± std
- `WF_summary.json` — 통합 요약
- `WF_comparison.png` — fold별 비교 그림
