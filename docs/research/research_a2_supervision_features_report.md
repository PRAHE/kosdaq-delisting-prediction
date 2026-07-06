# 외부 데이터 (A2) — 관리종목 지정 이력 추가

## 강한 신호이지만 A1 감사의견과 **중복** — 한계 효용 0

작성일: 2026-05-31
작성자: 이현지
관련 계획: `docs/다음 실험 설계안(모델 개선).md` §A2
선행 연구: S0 / SR / S1 walk-forward / A3 시장 / A1 감사
재현 코드: `src/research/a2_supervision/`, `src/research/s1_irregular_ts/`
결과 산출물: `results/research_s1_walkforward_market_audit_supervision/`

---

## 초록 (Abstract)

A1 감사의견 추가까지의 walk-forward 평균 PR-AUC는 0.408 (Transformer +
시장 + 감사, 44 features)로 S0 baseline의 95% Bayesian CI [0.180, 0.421]
상단 가장자리에 도달했다. 본 보고서는 외부 데이터의 세 번째 stream —
**A2 KRX 거래소공시 (관리종목·투자주의·거래정지) 이력**을 추가한 결과를
보고한다.

데이터: DART OpenAPI `list.json` 엔드포인트 + `pblntf_ty='I'` 거래소공시
필터로 fixed_v1 패널 1,536 corps × 2014~2024년 수집. **report_nm 키워드
매칭**으로 관리종목/투자주의환기/투자경고/거래정지/단기과열 관련 공시 추출
— 총 **655개 이벤트** 확보, 실패 0건. 분기 말 시점 정합성으로 6개 피처
생성: `is_supervised_now`, `days_since_last_concern`,
`n_supervision_events_5y`, `n_concern_events_5y`, `has_trading_halt_3y`,
`has_any_supervision_history`.

세 가지 결과가 **명확한 negative result**를 보여준다.

**(1)** Transformer test PR-AUC: **0.408 → 0.402** (−0.006). 4/4 fold 중 3개
에서 미세 하락, 1개에서 미세 상승 — 사실상 **동등**. P@20도 0.650 → 0.621
(−0.029)로 약간 손실.

**(2)** GRU에서는 미세 개선 (0.346 → 0.360, +0.014). 표준편차는 Transformer
에서 0.046 → 0.040으로 좁아짐 — **학습 안정성 약간 개선**, 그러나 평균 성능
향상은 없음.

**(3)** 단순 교차표가 그 이유를 직접 설명한다. `is_supervised_now=1` 행에서
양성 비율은 5.67% (base rate 0.46% 대비 **12배 ↑**) — 의미 있는 신호이나
A1 감사의견 (`audit_nonclean_consec=2` → 96배 ↑) 대비 **약 8배 약함**.
즉 **A1이 이미 A2의 신호 대부분을 포함**.

해석: KRX 코스닥 상장규정 28조~38조에서 비적정 감사의견과 관리종목 지정
사유는 **같은 부실 신호 트리거**를 공유한다. A1을 이미 가진 상태에서 A2의
한계 효용은 거의 0이다.

종합: 본 결과는 외부 데이터 우선순위에 대한 명확한 가이드다 — **A1
감사의견 우선, A2는 보조 (필요 시)**. 또한 이는 **데이터 통합의 일반 원칙**
을 보여준다 — *같은 source(KRX 부실 표지)에서 나온 데이터는 한 변수 추가
후 두 번째 변수는 한계 효용 ↓*.

---

## 1. 배경: A1 결과 후 A2의 기대치

A1 감사의견 추가는 walk-forward 평균 PR-AUC **0.289 → 0.408** (+0.119)
의 큰 도약을 만들었고, 단일 fold 4 (2023 valid)는 0.444로 S0 baseline CI
[0.180, 0.421] **완전 돌파**. 그러나:

- 평균 0.408 vs CI 상단 0.421 — **여전히 CI 내 가장자리**
- **운영 지표 P@20 = 0.650** — top 20 중 13개 양성 (good but improvable)
- 남은 양성 영역 정밀도 향상이 필요

A2 관리종목은 이론상 가장 직접적 상폐 선행 신호다 (코스닥 상폐 기업 80%+
가 관리종목 경험). A1과 결합 시 추가 +0.05~0.10 PR-AUC 개선이 가능할지를
검증한다.

---

## 2. 데이터 수집

### 2.1 DART 거래소공시 API

- 엔드포인트: `https://opendart.fss.or.kr/api/list.json`
- 파라미터: `corp_code`, `bgn_de=20140101`, `end_de=20241231`,
  `pblntf_ty='I'` (거래소공시)
- 대상: 1,506 corps × paginated 응답 (대부분 1 페이지)
- 응답 필드: `rcept_no`, `rcept_dt` (YYYYMMDD), `report_nm` (공시명), `stock_code`

### 2.2 키워드 필터링

`report_nm`에 다음 중 하나 포함된 공시만 추출:

| 키워드 | 의미 |
|--------|------|
| `관리종목` | 관리종목 지정/해제/우려 |
| `투자주의환기` | 투자주의환기종목 지정/해제 |
| `투자경고` | 투자경고종목 |
| `투자위험` | 투자위험종목 |
| `투자주의종목` | 일반 투자주의 |
| `단기과열` | 단기과열종목 (시장경보) |

### 2.3 수집 결과

| 항목 | 값 |
|------|-----|
| API 호출 corps | 1,506 (캐시된 30개 skip 후) |
| 실패 | **0건** |
| 추출된 이벤트 총수 | **655건** |
| ≥1 이벤트 corp 수 | **181 / 1,536** (11.8%) |
| Wall time (4 workers) | **~30분** |

캐시: `data/supervision/{corp_code}.json` (1,536개 파일)

---

## 3. 분기별 6개 피처 산출

### 3.1 이벤트 분류

`report_nm` 키워드로 6 카테고리 분류:

- **DESIGNATION**: 관리종목 지정 (실제 지정)
- **RELEASE**: 관리종목 지정해제
- **CONCERN**: 관리종목 지정 우려 (사전 경고)
- **CAUTION**: 투자주의환기종목 지정
- **TRADING_HALT**: 주권매매거래정지
- **OTHER**: 기타 (단기과열 등)

### 3.2 시점 정합성

공시는 발생 시점(`rcept_dt`)에 즉시 공개되므로 (stock, year=Y, quarter=Q)의
**분기 말 시점까지**의 공시만 사용. 미래 공시는 입력하지 않음.

분기 시점 매핑: Q1=3월말, H1=6월말, Q3=9월말, ANNUAL=12월말.

### 3.3 6개 피처

| 피처 | 정의 |
|------|------|
| `is_supervised_now` | 시점 t에서 관리종목 상태 (1=마지막 DESIGNATION 이후 RELEASE 없음) |
| `days_since_last_concern` | 가장 최근 CONCERN/DESIGNATION/CAUTION 후 일수 (없으면 9999) |
| `n_supervision_events_5y` | 직전 5년간 DESIGNATION/RELEASE/CONCERN 이벤트 수 |
| `n_concern_events_5y` | 직전 5년간 CONCERN 이벤트 수 |
| `has_trading_halt_3y` | 직전 3년 내 거래정지 경험 (1/0) |
| `has_any_supervision_history` | 어떤 형태의 부실 공시든 존재 (1/0) |

### 3.4 라벨과의 단순 교차표

**train 라벨 × `is_supervised_now`**:

| is_supervised_now | 음성 | 양성 | 양성 비율 |
|-------------------|------|------|----------|
| 0 (정상) | 33,030 | 152 | **0.46%** (base) |
| 1 (관리종목) | 2,180 | 131 | **5.67%** (12배 ↑) |

A1의 비적정 연속 2회 → 96배 ↑와 비교하면 **약 8배 약한 신호**.

**train 라벨 × `has_any_supervision_history`**:

| has_history | 음성 | 양성 |
|-------------|------|------|
| 0 | 32,808 | 149 |
| 1 | 2,402 | 134 |

상폐 양성 283개 중 47%(134개)가 history 있음 — 의미 있음, 그러나 53%는
관리종목 경험 없이 상폐.

---

## 4. 학습 결과 — Walk-Forward 4 fold × 3 seed

### 4.1 전체 평균 (12 측정)

| 모델 / 설정 | Test PR-AUC | std | ROC-AUC | P@20 | P@50 |
|------------|-------------|-----|---------|------|------|
| Transformer + 시장 + 감사 (44 feat) — **A1까지** | **0.408** | 0.046 | 0.919 | **0.650** | **0.440** |
| Transformer + 시장 + 감사 + 관리종목 (50 feat) — **A1+A2** | 0.402 | **0.040** | 0.920 | 0.621 | 0.408 |
| Δ | **−0.006** | −0.006 | +0.001 | −0.029 | −0.032 |
| GRU + 시장 + 감사 (44 feat) | 0.346 | 0.059 | 0.913 | 0.529 | 0.413 |
| GRU + 시장 + 감사 + 관리종목 (50 feat) | **0.360** | 0.074 | 0.912 | **0.579** | 0.417 |
| Δ | **+0.014** | +0.015 | −0.001 | +0.050 | +0.004 |

### 4.2 Fold별 Transformer (A1 vs A1+A2)

| Fold | Valid | A1만 (44) | A1+A2 (50) | Δ |
|------|-------|-----------|------------|---|
| 1 | 2020 | 0.353 ± 0.058 | **0.356 ± 0.033** | +0.003 |
| 2 | 2021 | 0.412 ± 0.018 | 0.397 ± 0.032 | −0.015 |
| 3 | 2022 | 0.423 ± 0.030 | 0.418 ± 0.034 | −0.005 |
| 4 | 2023 | 0.444 ± 0.018 | 0.438 ± 0.013 | −0.006 |

**3/4 fold 미세 하락, 1 fold 미세 상승. 사실상 동등.**
fold별 std는 일부 감소 (1, 4 fold에서) — 학습 안정성 마진 개선.

### 4.3 P@20 — 일부 손실

| Fold | A1만 | A1+A2 | Δ |
|------|------|-------|---|
| 1 | 0.600 | 0.533 | **−0.067** |
| 2 | 0.617 | 0.600 | −0.017 |
| 3 | 0.700 | 0.667 | −0.033 |
| 4 | 0.683 | 0.683 | 0 |

특히 fold 1에서 P@20이 6.7% 감소 — top 20 중 1.3명을 놓치는 효과.

---

## 5. 정직한 해석 — A1과 A2의 중복

### 5.1 KRX 부실 표지 시스템은 한 소스

코스닥 상장규정 28조~38조에 따라 관리종목 지정과 비적정 감사의견은 **같은
부실 트리거**를 공유한다:

| 항목 | A1 (감사의견) | A2 (관리종목) |
|------|---------------|----------------|
| 38조 트리거 | 비적정 2년 연속 → 상폐 사유 | 38조 사유 충족 시 관리종목 |
| 28조 트리거 | — | 영업손실 4년, 자본잠식 등 |
| 정보 source | 회계감사인 | 거래소 자동 판정 |
| 공시 시점 | 사업보고서 (회계연도 +3개월) | 사유 발생 직후 |

대부분의 관리종목 지정은 **비적정 감사의견이 그 원인 중 하나**이므로,
A1 (감사의견 + 누적) 변수가 이미 A2 (관리종목)의 정보 대부분을 포함한다.

### 5.2 단순 신호 강도 비교

| 트리거 | 양성 비율 | base rate 대비 |
|--------|-----------|----------------|
| `audit_nonclean_consec = 2` (A1) | **38.5%** | **96배 ↑** |
| `is_supervised_now = 1` (A2) | 5.67% | 12배 ↑ |
| `audit_opinion_t = 3` (의견거절) | 30.6% | 76배 ↑ |
| `has_trading_halt_3y = 1` (A2) | (거의 0, sample 부족) | — |

**A1이 A2보다 약 8배 강한 신호**. A1 추가 시 모델은 A1에 큰 가중치를
부여하고, A2의 추가 입력은 noise처럼 작용.

### 5.3 GRU가 A2로 약하게 개선되는 이유

| 모델 | A1만 (44) | A1+A2 (50) | Δ |
|------|-----------|------------|---|
| GRU | 0.346 | **0.360** | +0.014 |
| Transformer | 0.408 | 0.402 | −0.006 |

GRU는 단순 모델이라 A1을 완벽히 활용하지 못함 → A2가 보완 역할. Transformer
는 A1을 이미 잘 활용해서 A2의 추가 가치 거의 0.

P@20 측면에서는 GRU의 개선이 더 명확 (0.529 → 0.579, +0.050) — A2가
GRU의 양성 ranking을 보강.

### 5.4 학술적 가치 (negative result)

본 결과는 **학술적으로 의미 있는 발견**이다:

> "한국 코스닥 데이터에서 KRX 부실 표지 시스템의 두 정보원(감사의견,
> 관리종목)은 강하게 중복되며, 둘을 함께 사용해도 PR-AUC는 더 오르지
> 않는다. 첫 번째 정보원이 두 번째의 한계 효용을 거의 0으로 만든다."

이는 **외부 데이터 통합의 일반 원칙**에 대한 실증 — 같은 source family에서
나온 변수들의 marginal contribution은 빠르게 감소한다.

---

## 6. 한계 및 후속 가설

- **3 seed로 std 추정** (S1과 동일). 더 robust한 평가에는 5+ seed 권장.
- **A2의 신호가 약한 이유는 cover rate** — 1,536 corps 중 181개(11.8%)만
  공시 이벤트 보유. 나머지 88.2%는 모두 NaN/0. 코스닥 부실 80%+가 관리
  종목 경험이라는 이론과의 괴리 — 키워드 매칭 누락 가능성 검토 필요.
  > **⚠ 사후 검증 (2026-06-21, `docs/research_supervision_v3_check_report.md` #7):
  > 본 negative result는 불완전 수집에 의한 confounded 결론이다.** 키워드가
  > "관리종목" 글자에 의존해 상폐 파이프라인의 가장 직접적 공시(상장적격성
  > 실질심사·매매거래정지·상장폐지 사유 발생·정리매매)를 전부 놓쳤다. 누락된
  > 상폐기업 37개를 원본 재수집한 결과 **확장 키워드로 전부 복구 → 상폐기업
  > 실제 coverage는 11.8%/65%가 아니라 100%**다. 따라서 "A1과 중복·한계효용 0"은
  > 신뢰할 수 없으며, **확장 키워드 재수집 + 엄격한 시점 컷오프(정리매매 등
  > 말기 공시 배제, 누수 방지)** 후 재평가해야 한다.
  >
  > **→ 재평가 완료 (2026-06-21, `docs/research_a2_reeval_report.md`):** 전 종목
  > 무필터 재수집(212,302건) + v2 피처(조기/말기 분리)로 재평가한 결과, **train
  > 양성의 81.6%가 조기 관리종목 신호 보유**. 감사 위에 조기 신호를 더하면
  > **ROC 0.870→0.938, P@20 0.60→0.75로 개선**(ranking·top-K). 단 PR-AUC marginal은
  > +0.022로 비유의(감사와 source 중복 + 양성 56 저표본). PR-AUC를 유의하게
  > 올리는 말기 신호(정리매매·상폐결정)는 누수라 배제.
  > **딥모델 재평가:** best 모델(Transformer+재무+감사, T-38 0.458)에 조기 관리
  > 종목을 시퀀스로 추가하면 **PR 0.458→0.426·P@20 0.75→0.70으로 약손해**(ROC만
  > +0.027). 강한 모델은 재무+감사에서 같은 신호를 이미 추출 → 중복.
  > **최종 결론: A2의 "관리종목은 운영 best 모델에 불필요"는 — 데이터 수집은
  > 불완전했으나(#7) — 정정된 이유(강한 모델의 신호 중복)로 유지된다.** 관리종목
  > 이득은 약한 RF 스냅샷에서만 나타난다.
- **`has_trading_halt_3y`가 train 양성에서 거의 0** — 거래정지 공시가
  사전 시점에서 잡히지 않음 (라벨 = 상폐 1년 전, 거래정지는 그 이후).
  → 라벨 누수 없는 boundary로 의도된 결과이긴 함.
- **fold별 std가 일부 감소** (Transformer fold 1, 4) — 데이터 noise를 살짝
  흡수했을 가능성. PR-AUC 평균은 동등.
- **검토 가치**: A2에서 `is_supervised_now`만 단일 피처로 추가 (다른 5개
  제외) → 더 단순한 신호 정제. A1이 이미 강하므로 큰 변화 기대 어렵지만.

---

## 7. 결정 — 현재 best baseline 유지

| 후보 | PR-AUC | P@20 | 권장도 |
|------|---------|------|--------|
| Transformer + 시장 + 감사 (44 feat) | **0.408** | **0.650** | **best** |
| Transformer + 시장 + 감사 + 관리종목 (50 feat) | 0.402 | 0.621 | 동등, 추가 가치 미미 |
| GRU + 시장 + 감사 + 관리종목 (50 feat) | 0.360 | 0.579 | A1 단독 대비 GRU에서만 약한 개선 |

**현재 stream의 best baseline = Transformer + 시장 + 감사 (44 features)**.
Walk-forward 평균 PR-AUC 0.408, P@20 0.650, ROC-AUC 0.919.

A2는 보고서 가치 (negative result + 학술적 발견)는 있지만 운영 모델에는
포함하지 않음을 권장.

---

## 8. 다음 단계 (갱신)

| 항목 | 결과 | 우선순위 |
|------|------|---------|
| A1 감사의견 ✅ | +0.119 PR-AUC (큰 도약) | **사용** |
| A2 관리종목 ✅ | ±0 (A1과 중복) | **보조 (선택)** |
| A3 시장 ✅ | +0.008 PR-AUC, ROC +0.033 | 사용 (ranking 개선) |
| **Hyperparameter sweep** | 미수행 | **권장 (다음)** |
| Stream 3 hazard 모형 | 미수행 | A1+A3와 결합 검토 |
| Phase 4 Neural CDE | 미수행 | GPU 확보 시 |
| **NaN ratio mask channel** | 진단 완료 | 신호 약함 → 미권장 |

---

## 9. 재현성

```powershell
# 1) 관리종목 공시 수집 (~30분)
.venv\Scripts\python.exe -m src.research.a2_supervision.fetch_supervision --workers 4

# 2) 분기별 6개 피처 산출 (~1분)
.venv\Scripts\python.exe -m src.research.a2_supervision.supervision_features

# 3) Walk-forward (시장+감사+관리종목, ~90분)
.venv\Scripts\python.exe -m src.research.s1_irregular_ts.run_s1_walkforward `
    --seeds 42,7,13 --epochs 25 --with-market --with-audit --with-supervision
```

---

## 10. 산출물

`src/research/a2_supervision/`:
- `fetch_supervision.py` — DART 거래소공시 수집 (`pblntf_ty='I'` + 키워드)
- `supervision_features.py` — 분기별 6개 피처 (event timeline → quarterly)

`data/supervision/` — 1,536 corp별 이벤트 JSON
`preprocess/data/supervision/supervision_features.csv` — 45,854 행 × 9 컬럼

`results/research_s1_walkforward_market_audit_supervision/`:
- `WF_grid.csv`, `WF_fold_summary.csv`, `WF_summary.json`, `WF_comparison.png`

---

## 11. 참고 문헌

- KRX 코스닥 상장규정 제28조~제38조 (관리종목 지정 및 상장폐지 사유).
- DART OpenAPI 공시검색 (`list.json`) — `pblntf_ty='I'` 거래소공시.
- Geiger, M.A. & Raghunandan, K. (2002). *Going-Concern Opinions and the
  Prediction of Corporate Failure.* Auditing — A1과 A2 신호 중복 이유.
