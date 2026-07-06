# 모델군 확장 — 스냅샷·시계열 다른 모델들 (같은 조건)

## 외부데이터 이득은 RF 특화인가? best 피처셋에서 최선의 아키텍처는?

작성일: 2026-06-21
작성자: 이현지
선행 연구: 점검 스위트 #1~#8b / A2 재평가
재현 코드: `src/research/rf_feature_control/run_snapshot_models.py`,
`run_sequence_models.py`, `src/research/s1_irregular_ts/lstm_baseline.py`, `tcn.py`
결과 산출물: `results/research_model_family/`

---

## 초록 (Abstract)

점검에서 스냅샷 모델은 RF, 시계열은 Transformer로 굳어졌고 천장 돌파의 동력은
감사 피처였다. 그러나 **그 이득이 RF 특화인지**, **best 피처셋(38=재무+감사)에서
다른 아키텍처가 더 나은지**는 미검증이었다. 본 실험은 **완전히 동일한 조건**
(같은 스냅샷/시퀀스·signed_log1p·단일 split·5-seed ensemble·paired Bayesian
bootstrap)에서 모델만 교체해 두 모델군을 비교한다.

**(A) 스냅샷 tabular 5종**(RF·XGB·LightGBM·HistGB·LogReg):
- **감사 이득(33→38)은 모델 무관하게 일반화** — 점추정 전부 양수, 5개 중 4개
  유의(RF +0.107, XGB +0.108, LGBM +0.117, LogReg +0.139; HistGB +0.038 비유의).
  **"외부데이터가 좋아진 것"은 RF 특화가 아니라 데이터-driven**임이 확정.
- **RF를 유의하게 능가하는 tabular 모델은 없음.** 오히려 **LogReg가 RF와 동등
  이상**(PR 0.413 vs 0.367, P@20 **0.75**, Δ vs RF +0.045 비유의) — 신호가 상당히
  **선형 분리 가능**함을 시사. 부스팅(XGB/LGBM/HistGB)은 양성 56 극불균형에서
  RF보다 **유의하게 열위**.

**(B) 시계열 sequence 4종**(GRU·LSTM·TCN·Transformer+T2V, 모두 38feat):
- **Transformer가 최고 점추정(0.458)이나 GRU(0.415)·TCN(0.429)과 통계적 유의차
  없음**(CI 0 포함). 즉 합리적 아키텍처 간 PR-AUC는 본 표본에서 구분 불가.
- **유일한 유의차: LSTM이 GRU보다 유의하게 열위**(Δ −0.040, P=0.012) —
  소데이터에서 LSTM의 추가 파라미터가 약손해.

**(C) 군 간 비교**(최강 스냅샷 LogReg-38 vs 최강 시계열 Transformer-38):
- 단일 split에선 Transformer−LogReg=+0.045 **비유의**였으나, **4-fold walk-forward
  (검정력 4배)에서 +0.060, CI [+0.004, +0.127], P=0.984로 유의**(4 fold 전부 양수).
  즉 시계열은 *최강* 스냅샷(LogReg) 대비로도 진짜 이득이며, 단일 split의 비유의는
  저표본 artifact였다.

종합: **계층적 결론** — (i) 군 *내부* 알고리즘은 무차별(RF≈LogReg, GRU≈TCN≈
Transformer; LSTM·부스팅만 열위), (ii) 그러나 *아키텍처 클래스*(스냅샷→시계열)는
walk-forward에서 **유의(+0.060)**, (iii) 기여 크기 **감사 피처(+0.11) > 시계열
모델링(+0.06) > 군 내부 알고리즘(≈0)**. best는 시계열 Transformer(시장 제외 38feat,
WF 0.446); 단순 LogReg(0.387)는 최강 스냅샷으로서 RF와 동급.

---

## 1. 공통 조건

| 항목 | 내용 |
|------|------|
| 데이터 | `prepare_phase1_datasets` (fixed_v1, test 2024 양성 56) |
| 피처셋 | 33(재무) / 38(재무+감사). 시장 제외(#8 frozen artifact) |
| 전처리 | median impute(train fit) + signed_log1p |
| split | 단일 (train 2015–22 / valid 23 / test 24) |
| seed | [42,7,13,21,100], 5-seed 확률 ensemble |
| 라벨 | 스냅샷 = L1, 시계열 = L3 train / L1 eval |
| 평가 | 단일 모델 CI = Bayesian bootstrap; 모델 간 = **paired** Bayesian bootstrap (n=5,000) |
| 튜닝 | 고정 default (모델별 Optuna 없음) |

---

## 2. Part A — 스냅샷 tabular 모델군

### 2.1 모델별 PR-AUC (33 → 38, 5-seed ensemble)

| 모델 | PR(33) | PR(38) | ROC(38) | P@20(38) |
|------|--------|--------|---------|----------|
| **RF** | 0.258 | 0.367 | 0.870 | 0.60 |
| XGBoost | 0.180 | 0.294 | 0.872 | 0.55 |
| LightGBM | 0.184 | 0.303 | 0.901 | 0.60 |
| HistGB | 0.162 | 0.200 | 0.831 | 0.40 |
| **LogReg** | 0.272 | **0.413** | 0.874 | **0.75** |

### 2.2 감사 이득 (38 − 33, paired bootstrap)

| 모델 | Δ | 95% CI | P | 유의 |
|------|-----|--------|---|------|
| RF | +0.107 | [+0.040, +0.180] | 0.999 | ✓ |
| XGBoost | +0.108 | [+0.046, +0.174] | 0.999 | ✓ |
| LightGBM | +0.117 | [+0.053, +0.189] | 1.000 | ✓ |
| HistGB | +0.038 | [−0.013, +0.094] | 0.927 | ✗ |
| LogReg | +0.139 | [+0.077, +0.215] | 1.000 | ✓ |

→ **점추정 전부 양수, 4/5 유의. 감사 이득은 모델군 전반에 일반화** — RF 특화 아님.

### 2.3 RF-38 대비 (model_38 − RF_38, paired bootstrap)

| 모델 | Δ vs RF | 95% CI | 판정 |
|------|---------|--------|------|
| XGBoost | −0.072 | [−0.140, −0.010] | 열위 |
| LightGBM | −0.063 | [−0.130, −0.001] | 열위 |
| HistGB | −0.164 | [−0.245, −0.094] | 열위 |
| LogReg | +0.045 | [−0.015, +0.116] | 동등 |

→ **RF를 유의하게 능가하는 모델 없음.** LogReg만 RF와 동등(점추정은 더 높음).
부스팅 3종은 모두 RF보다 유의 열위.

### 2.4 해석
- **감사 신호는 모델 종류와 무관한 데이터 효과** — 사용자 질문("RF 특화인가")에
  대한 답: **아니다, 일반화된다.**
- **LogReg가 RF와 동급(P@20 0.75로 오히려 최고)** → 재무+감사 신호가 대체로
  선형적(의견거절·frozen 등 강한 단조 신호). 복잡한 트리/부스팅이 이점을 못 줌.
- **부스팅 열위**는 S0의 exp_012(XGB)·exp_015(LGBM) 결과와 일관 — 양성 56개에서
  부스팅이 RF/선형보다 과적합·불안정.

---

## 3. Part B — 시계열 sequence 모델군 (38feat)

### 3.1 모델별 (5-seed ensemble)

| 모델 | PR-AUC | 95% CI | ROC | P@20 | P@50 |
|------|--------|--------|-----|------|------|
| GRU | 0.415 | [0.286, 0.551] | 0.862 | **0.80** | 0.46 |
| LSTM | 0.371 | [0.246, 0.516] | 0.862 | 0.75 | 0.42 |
| TCN | 0.429 | [0.306, 0.560] | **0.912** | 0.65 | **0.50** |
| **Transformer+T2V** | **0.458** | [0.337, 0.585] | 0.905 | 0.75 | 0.44 |

(Transformer 0.458은 #8b의 T-38과 일치 — 조건 동일성 확인.)

### 3.2 paired Δ

| 대조 | Δ | 95% CI | P | 판정 |
|------|-----|--------|---|------|
| LSTM − GRU | −0.040 | [−0.085, −0.005] | 0.012 | **GRU 유의 우위** |
| TCN − GRU | +0.012 | [−0.052, +0.072] | 0.658 | = |
| Transformer − GRU | +0.040 | [−0.050, +0.132] | 0.817 | = |
| LSTM − Transformer | −0.081 | [−0.175, +0.010] | 0.042 | ≈ (경계) |
| TCN − Transformer | −0.028 | [−0.081, +0.015] | 0.113 | = |

### 3.3 해석
- **Transformer가 최고 점추정(0.458)이지만 GRU·TCN과 통계적으로 구분 안 됨**
  (CI 모두 0 포함). 합리적 sequence 아키텍처 간 PR-AUC 차이는 양성 56개로는
  검출 불가.
- **유일한 유의차: LSTM < GRU**(−0.040, P=0.012). LSTM은 GRU보다 게이트·파라미터가
  많아 소데이터에서 약손해 — Phase 2 GRU-D negative result와 같은 결("복잡도↑가
  본 규모에서 불리").
- TCN은 ROC(0.912)·P@50(0.50) 최고로 운영 ranking에서 경쟁력 있으나 PR-AUC는
  Transformer와 무차별.

---

## 4. 종합 — 두 모델군을 관통하는 결론

1. **모델 선택은 PR-AUC 유의차를 거의 만들지 못한다.** 스냅샷에서도(RF≈LogReg,
   부스팅만 열위), 시계열에서도(Transformer≈GRU≈TCN, LSTM만 열위) — 양성 56개
   표본에서 "합리적 모델군 내" 차이는 noise band 안. **피처(감사)가 본질**이라는
   점검 결론을 모델 축에서 재확인.
2. **기존 선택 견고:** 스냅샷 RF, 시계열 Transformer는 각 군에서 동급 최강.
   교체 이득 없음.
3. **피해야 할 선택:** tabular 부스팅(XGB/LGBM/HistGB, RF 대비 유의 열위),
   sequence LSTM(GRU 대비 유의 열위).
4. **부수 발견:** **LogReg(스냅샷)** 가 RF와 동등(P@20 0.75) — 신호의 선형성.
   단순·해석가능 모델이 강력하다는 점은 포트폴리오 서사로 가치.

### 4.1 군 간 비교 — 최강 스냅샷 vs 최강 시계열 (마무리)

위 결론을 닫기 위해 군을 가로질러 paired 검정을 수행했다(동일 test 행, 38feat).

| 모델 | PR-AUC | 95% CI | P@20 |
|------|--------|--------|------|
| RF-38 (스냅샷) | 0.367 | [0.252, 0.494] | 0.60 |
| LogReg-38 (스냅샷, 최강) | 0.413 | [0.291, 0.546] | 0.75 |
| Transformer-38 (시계열, 최강) | 0.458 | [0.337, 0.585] | 0.75 |

| paired Δ | Δ | 95% CI | P | 판정 |
|----------|-----|--------|---|------|
| Transformer − RF | +0.090 | [+0.028, +0.160] | 0.998 | ✓ 유의 (#8b 재현) |
| **Transformer − LogReg** | **+0.045** | **[−0.024, +0.116]** | 0.904 | **= 동등** |
| LogReg − RF | +0.045 | [−0.015, +0.116] | 0.923 | = 동등 |

→ 단일 split에서는 Transformer−LogReg=+0.045가 **비유의**(CI 0 포함)였다. 그러나
이는 양성 56개의 저표본 검정력 때문일 수 있어 walk-forward로 재검증했다(§4.2).

### 4.2 군 간 walk-forward 재검증 — 단일 split을 뒤집다

#3처럼 4-fold walk-forward(측정 4배)로 fold-평균 paired Δ posterior를 산출했다
(`run_cross_family_wf.py`, Transformer 3-seed, n_boot=3,000).

| fold | RF | LogReg | Transformer |
|------|------|--------|-------------|
| 1 (train 2015–19) | 0.372 | 0.325 | 0.398 |
| 2 (train 2015–20) | 0.263 | 0.399 | 0.447 |
| 3 (train 2015–21) | 0.379 | 0.410 | 0.459 |
| 4 (train 2015–22) | 0.366 | 0.413 | 0.479 |
| **wf 평균** | **0.345** | **0.387** | **0.446** |

| fold-평균 paired Δ | Δ | 95% CI | P | 판정 |
|--------------------|-----|--------|---|------|
| **Transformer − LogReg** | **+0.060** | **[+0.004, +0.127]** | 0.984 | **✓ 유의** |
| Transformer − RF | +0.100 | [+0.033, +0.173] | 0.999 | ✓ 유의 |
| LogReg − RF | +0.041 | [−0.018, +0.104] | 0.918 | = 동등 |

fold별 Transformer−LogReg: +0.076 / +0.049 / +0.048 / +0.065 — **4 fold 전부 양수**.

→ **검정력을 4배로 올리면 Transformer가 최강 스냅샷(LogReg)도 유의하게 능가한다**
(+0.060, CI가 0을 가까스로 배제, 4 fold 일관). 단일 split의 "동등(+0.045)"은
저표본 artifact였다. 즉 **시계열(시간적 모델링)의 가치는 비교 baseline이 RF든
LogReg든 실재**하며, #8b 결론이 최강 스냅샷 대비로 확장·강화된다.

### 함의 (정정된 계층 구조)
- **군 *내부* 모델은 무차별**: 스냅샷 RF≈LogReg, 시계열 GRU≈TCN≈Transformer
  (LSTM·부스팅만 열위). 특정 알고리즘 선택은 천장에 영향 없음.
- **그러나 *아키텍처 클래스*(스냅샷 → 시계열)는 유의**: 시간적 모델링이 최강
  스냅샷 대비로도 **+0.060(WF)** 의 진짜 이득. 단일 split에선 묻혔다 walk-forward
  검정력으로 드러남.
- 기여 크기 순서: **감사 피처(+0.11) > 시계열 모델링(+0.06) > 군 내부 알고리즘(≈0)**.
- 단순·해석가능 LogReg가 *최강 스냅샷*으로서 RF와 동급(0.387)이라는 점은 여전히
  유효하나, 최선은 시계열(Transformer 0.446)이다.

---

## 5. 재현성

```powershell
.venv\Scripts\python.exe -m src.research.rf_feature_control.run_snapshot_models   # Part A (~5분)
.venv\Scripts\python.exe -m src.research.rf_feature_control.run_sequence_models    # Part B (~25분)
```

신규 모델: `src/research/s1_irregular_ts/lstm_baseline.py`(LSTM),
`tcn.py`(TCN, Bai et al. 2018). 추가 의존성 없음(torch 2.12 + 기존). 산출물:
`results/research_model_family/{snapshot,sequence}_models_{summary.json,.csv}`.

---

## 6. 핵심 한 줄

> **감사 피처의 이득은 모델군 전반에 일반화된다(스냅샷 5종 모두 양의 audit-lift).
> 군 *내부* 알고리즘은 무차별(RF≈LogReg, GRU≈TCN≈Transformer; 부스팅·LSTM만 열위)
> 이나, *아키텍처 클래스*(스냅샷→시계열)는 walk-forward에서 유의하다 —
> Transformer가 최강 스냅샷(LogReg)도 +0.060(WF, 4 fold 일관)으로 능가. 기여 순서는
> 감사 피처(+0.11) > 시계열 모델링(+0.06) > 군 내부 알고리즘(≈0).**
