# 점검 실험 종합 개요 (Inspection Suite Overview)

## research_* 보고서 논리 점검에서 출발한 6개 검증 실험의 통합 요약

작성일: 2026-06-20
작성자: 이현지
대상: `docs/research_*_report.md` 전반의 인과 귀속·피처 정당성 점검
관련 코드: `src/research/rf_feature_control/`
관련 산출물: `results/research_rf_feature_control*/`, `results/research_audit_check/`

---

## 0. 배경과 동기

기존 research stream의 헤드라인(`best_ci`)은 **"단순 스냅샷 RF (PR-AUC 0.27) →
시계열 Transformer + 시장 + 감사 (0.45), paired Δ +0.18 → 천장 돌파"** 였다.
그러나 이 +0.18에는 **(a) 모델(RF→Transformer) · (b) 입력(스냅샷→K=3 시계열) ·
(c) 외부 피처(+시장+감사)** 세 변화가 한꺼번에 들어 있어 무엇이 진짜 동력인지
분리되지 않았다. 본 점검 스위트는 이를 분리하고, 각 피처의 정당성(누수·자명성·
artifact)을 검증한다. 총 6개 실험.

---

## 1. 공통 실험 설정 (모든 실험 공유)

| 항목 | 내용 |
|------|------|
| **데이터 패널** | `preprocess/data/processed_fixed_v1/fixed_N1/exp-A` (1,536종목 분기 패널) |
| **분할** | 단일 split: train 2015–2022 / valid 2023 / **test 2024**. walk-forward: 4 fold (train cutoff × valid year, test 항상 2024) |
| **test 규모** | N=5,311, 양성 56 (fixed_N1, base rate 1.05%) |
| **전처리** | `SimpleImputer(median)` (train fit) + `signed_log1p`. 모든 변종 동일 |
| **피처 구성** | 재무 33 / +감사 5 = 38 / +시장 6 = 39 / 전체 44 |
| **RF (스냅샷)** | 시퀀스의 **마지막 timestep = 현재연도 스냅샷**. canonical 파라미터(n_estimators 200, max_depth 10, min_samples_leaf 5, max_features sqrt). 학습 라벨 = L1 fixed_N1 |
| **Transformer (시계열)** | Time2Vec + 1-layer encoder (dim 64, heads 4, dropout 0.2), K=3 시퀀스, focal loss(γ2 α.25), lr 5e-4. 학습 라벨 = L3 rolling_H24 |
| **평가 라벨** | 항상 L1 fixed_N1 (모든 보고서의 공정 비교 원칙) |
| **불확실성** | 5-seed(또는 3-seed) ensemble 확률 + **paired Bayesian bootstrap** (Dirichlet, n=3,000~5,000, 같은 가중치로 두 모델 동시 평가) |

**핵심 설계 포인트:** RF와 Transformer가 **동일 test 행·동일 전처리·동일 피처**를
공유하므로, 둘의 차이는 오직 "시계열 사용 여부"로 귀속된다. 이것이 인과 분리의
열쇠다.

---

## 2. 실험별 요약

### #1 — RF 피처 대조군 (단일 split)
- **데이터셋:** 단일 split, test 2024 (양성 56). 피처 33/38/39/44.
- **모델:** RF(스냅샷, 5-seed) 4변종 + Transformer(시계열) 2변종(33, 44).
- **방법:** 각 변종 5-seed ensemble PR-AUC + paired bootstrap으로 효과 분리.
- **결과:**

  | 변종 | PR-AUC | ROC | P@20 |
  |------|--------|-----|------|
  | RF-33 (재무) | 0.258 | 0.842 | 0.50 |
  | RF+audit (38) | 0.367 | 0.870 | 0.60 |
  | RF+market (39) | 0.407 | **0.988** | 0.35 |
  | RF-44 | 0.413 | 0.982 | 0.55 |
  | T-33 | 0.363 | 0.894 | 0.75 |
  | T-44 | 0.454 | 0.941 | 0.75 |

  paired Δ: 외부피처 전체(RF-44−RF-33) **+0.156** ✓ / 감사 단독 +0.107 ✓ /
  헤드라인(T-44−RF-33) **+0.195** ✓ (best_ci +0.18 재현) /
  **시계열 순효과(T-44−RF-44) +0.039, CI[−0.040,+0.119], P=0.83 ✗ 비유의**.
- **결론:** 헤드라인의 **80%가 외부 피처**. 스냅샷 RF에 44피처만 줘도 0.41.
  시계열의 추가 PR-AUC는 (시장 포함 조건에서) 비유의. → `research_rf_feature_control_report.md`

### #3 — RF walk-forward 대조군
- **데이터셋:** 4-fold walk-forward (train 2015–19/…/–22, valid 2020–23), test 2024.
- **모델:** RF / Transformer × {33, 44} × fold × 3-seed(42,7,13).
- **방법:** 모든 보고서가 "미산출"로 남긴 **RF의 fold별 test** 산출 + fold-평균
  paired Δ posterior(검정력 4배).
- **결과 (fold별 test PR-AUC ensemble, wf 평균):**

  | 모델 | f1 | f2 | f3 | f4 | **평균** |
  |------|----|----|----|----|------|
  | RF-33 | 0.230 | 0.203 | 0.235 | 0.258 | 0.231 |
  | T-33 | 0.238 | 0.294 | 0.331 | 0.367 | 0.307 |
  | **RF-44** | 0.425 | 0.324 | 0.426 | 0.421 | **0.399** |
  | T-44 | 0.372 | 0.424 | 0.438 | 0.460 | 0.423 |

  Δ_ts_base(T33−RF33) +0.076, CI[−0.0001,+0.153], P=0.975 (경계) /
  **Δ_ts_full(T44−RF44) +0.021, CI[−0.062,+0.105], P=0.69 ✗**. fold1은 RF-44가 우위(Δ−0.052).
- **결론:** **RF-44 wf 평균 0.399 ≈ A1의 천장돌파 증거 T-44 0.408** — "0.41 돌파"가
  시계열 없이 재현. 측정 4배로도 Δ_ts_full 비유의 → #1 확정. → 동 보고서 §10

### #2 — Audit-only 단독 모델
- **데이터셋:** 단일 split, test 2024.
- **모델:** RF(스냅샷) on **감사 5피처만**; 단일 피처 트리비얼 랭킹.
- **방법:** 감사 단독 PR-AUC를 RF-33(0.258)·RF+audit(0.367)과 대조.
- **결과:** RF audit-only **0.252** (≈ 재무 단독 0.258), ROC 0.717.
  trivial: `nonclean_consec` 0.227 / `opinion_t` 0.149.
- **결론:** 감사는 **단독으로는 천장을 못 깬다**(자명한 트리거 아님). +0.11
  기여는 재무와의 **상호작용**. → `research_audit_check_report.md`

### #4 — Leakage audit (감사 시점 정합성)
- **데이터셋:** test 양성 56개 × 감사 피처(`audit_features.csv`).
- **모델:** 없음 (시점 정합성 표 분석).
- **방법:** 각 양성의 "사용 의견 회계연도(year−1) ↔ 상폐연도(year+1)" gap 산출.
- **결과:** **gap = 2 회계연도 전수**(분포 {2년:56}) → 전향적 누수 아님.
  audit_observed 100%. **27/56(48%)만 이미 의견거절**, **29/56(52%)는 적정인데도
  상폐**(감사-blind, S0의 confident FN).
- **결론:** 감사는 누수 아님, 양성의 절반만 사전 포착. → 동 보고서

### #8 — 시장 피처 검증
- **데이터셋:** test 2024 × 시장 피처(`market_features.csv`, OHLCV 캐시).
- **모델:** 없음 (coverage·분포·거래일 진단).
- **방법:** 양성 vs 음성 coverage/분포 비교 + 마지막 거래일(stale/halt) 점검.
- **결과:** 양성 **44.6%가 frozen**(변동성≈0 & 거래량≈−100%), drawdown 결측
  60.7% — 음성은 각각 1.3%/1.4%. stale_days 중앙 2(날짜는 분기말 정합).
- **결론:** RF+market의 ROC 0.988은 **"죽은 주식 탐지기" artifact**. 조기 신호가
  아닌 상폐 임박 상태. → `research_market_check_report.md`

### #8b — frozen artifact 제거 후 시계열 vs 스냅샷
- **데이터셋:** test 2024 full vs **non-frozen**(94행 제외, 양성 31).
- **모델:** RF / Transformer × {38(시장 제외), 44} × 5-seed.
- **방법:** (A) 시장 피처 제거로 artifact 원천 제거, (B) non-frozen 한정 평가.
  각각 paired bootstrap T−RF.
- **결과:**

  | 모델 | full PR | non-frozen PR |
  |------|---------|----------------|
  | RF-38 | 0.367 | 0.269 |
  | **T-38** | **0.458** | 0.361 |
  | RF-44 | 0.413 | 0.340 |
  | T-44 | 0.454 | 0.368 |

  **시장 제거 38feat: T−RF = +0.090, CI[+0.029,+0.158], P=0.998 ✓ 유의** /
  38feat non-frozen +0.088 P=0.971(경계) / 44feat full +0.040 비유의.
- **결론:** **시장 artifact를 제거하면 시계열이 유의하게 우위.** "시계열 무용"은
  frozen crutch가 만든 착시. **전체 best = T-38(재무+감사, 시장 없음) = 0.458**.
  → `research_frozen_check_report.md`

### #7 — A2 관리종목 coverage 감사
- **데이터셋:** 패널 상폐기업 107개 + DART 'I' 공시 원본 재수집(무필터).
- **모델:** 없음 (수집 완전성 감사).
- **방법:** uncovered 상폐기업의 원본 공시를 재조회, 기존 키워드 vs 확장 키워드
  (상장폐지/실질심사/매매거래정지/정리매매)로 재분류.
- **결과:** 기존 coverage 70/107(65.4%) → **확장 키워드로 37개 전부 복구 →
  100%**, 진짜 부재 0. 기존 키워드는 "관리종목" 글자에 의존해 상폐 파이프라인의
  직접 공시(실질심사·매매거래정지·상장폐지)를 전부 놓쳤다.
- **결론:** **A2의 "관리종목 중복·무용"은 불완전 수집 confounded** — 확장
  키워드 재수집 + 시점 컷오프(말기 공시 배제, 누수 방지) 후 재평가 필요. →
  `research_supervision_v3_check_report.md`

### #9 — v3(L7) "valid 0.42 착시" 재현
- **데이터셋:** 단일 split. train=L7(v3), valid/test 두 정답(L1, L7).
- **모델:** RF(스냅샷) 5-seed.
- **방법:** 동일 valid 예측을 L1·L7 두 정답으로 평가해 base-rate 부풀림 정량화.
- **결과:** valid PR-AUC **L1 0.193 → L7 0.359 (+0.166)**, base rate 3.2배
  (39→123). test는 0.250→0.208(하락) — 착시는 valid 특이적.
- **결론:** SR §4.4 "평가 비일관성 착시" 가설 **재현 확인**. v3의 "0.42"는
  valid를 v3 라벨로 평가한 부풀림. → 동 보고서

### 모델군 확장 — 스냅샷·시계열 다른 모델들 (같은 조건)
- **데이터셋:** 33/38feat, 단일 split, 5-seed. **모델군:** 스냅샷 RF/XGB/LGBM/
  HistGB/LogReg, 시계열 GRU/LSTM/TCN/Transformer.
- **방법:** 동일 조건에서 모델만 교체 + paired bootstrap. 군 간 비교(최강 스냅샷
  LogReg vs 최강 시계열 Transformer)까지.
- **결과:** (A) 감사 이득은 **모델 무관 일반화**(스냅샷 5종 모두 양의 audit-lift,
  4/5 유의) — RF 특화 아님. RF 능가 모델 없음, **LogReg가 RF와 동등(P@20 0.75)**,
  부스팅 열위. (B) Transformer 최고 점추정(0.458)이나 GRU·TCN과 **유의차 없음**;
  **LSTM만 GRU보다 유의 열위**. (C) 군 간: 단일 split Transformer−LogReg=+0.045
  비유의였으나 **walk-forward에서 +0.060 유의**(4 fold 일관) → 시계열 우위는 최강
  스냅샷 대비로도 확정.
- **결론(계층적):** 군 *내부* 알고리즘은 무차별(RF≈LogReg, GRU≈TCN≈Transformer)이나
  *아키텍처 클래스*(스냅샷→시계열)는 WF에서 유의(+0.060). **기여 순서: 감사
  피처(+0.11) > 시계열(+0.06) > 군 내부 알고리즘(≈0).** 부스팅·LSTM은 회피.
  → `research_model_family_report.md`

---

## 3. 종합 결론

1. **천장 돌파(0.26→0.45)는 모델·시계열·피처의 혼재 효과였다.** 분리하면:
   - **외부 피처(특히 감사)** 가 1차 동력 — 스냅샷 RF만으로 0.40까지.
   - **시계열 모델링**도 진짜 기여 — 단, 시장 artifact를 걷어내야 드러난다.
2. **감사 피처는 정당하다.** 누수 아님(gap 2년), 자명한 트리거 아님(단독 0.25),
   양성의 48%만 포착 — 나머지 52%(적정 의견 상폐)는 재무 궤적이 필요한 영역.
3. **시장 피처는 함정이다.** 양성 45%가 frozen인 "죽은 주식" artifact. RF를
   허위로 부풀리고(0.367→0.413) Transformer엔 노이즈(0.458→0.454). **폐기 권장.**
4. **시계열의 PR-AUC 가치는 실재한다.** 시장 제거 시 T-38이 RF-38을 +0.090
   (P=0.998)으로 유의하게 능가. top-K 정밀도는 모든 조건에서 시계열 우위(P@20 0.75).
5. **정정된 best 모델 = Transformer + 재무 + 감사 (38 features, 시장 제외),
   test PR-AUC 0.458** — 기존 44feat(0.454)보다 높고, OHLCV 수집이 불필요한 가장
   단순한 외부데이터 구성.

### 최종 귀속 한 줄
> 천장 돌파의 정당한 동력은 **감사 피처 + 시계열 모델링**이다. 시장 피처는
> frozen artifact라 폐기하는 편이 낫고, 그 artifact가 그동안 "스냅샷이 시계열과
> 동률"이라는 착시를 만들었다. 정정된 best = Transformer + 재무 + 감사(0.458).

---

## 4. 산출물 맵

| 실험 | 코드 | 결과 폴더 | 보고서 |
|------|------|-----------|--------|
| #1 | `rf_feature_control/run_rf_control.py` | `results/research_rf_feature_control/` | `research_rf_feature_control_report.md` §1–9 |
| #3 | `rf_feature_control/run_wf_control.py` | `results/research_rf_feature_control_wf/` | 동 §10 |
| #2·#4 | `rf_feature_control/run_audit_check.py` | `results/research_audit_check/` | `research_audit_check_report.md` |
| #8 | `rf_feature_control/run_market_check.py` | `results/research_audit_check/` | `research_market_check_report.md` |
| #8b | `rf_feature_control/run_frozen_check.py` | `results/research_audit_check/` | `research_frozen_check_report.md` |
| #7 | `rf_feature_control/run_supervision_audit.py` | `results/research_audit_check/` | `research_supervision_v3_check_report.md` |
| #9 | `rf_feature_control/run_v3_recheck.py` | `results/research_audit_check/` | 동 보고서 |
| 모델군 | `rf_feature_control/run_snapshot_models.py`, `run_sequence_models.py`, `run_cross_family.py`, `run_cross_family_wf.py` (+ `s1_irregular_ts/lstm_baseline.py`, `tcn.py`) | `results/research_model_family/` | `research_model_family_report.md` |

**패치된 기존 보고서:** `best_ci`(귀속 콜아웃·§4.4·§5·§9), `a1_audit`(§3.4·§5),
`s1_walkforward`(§4.2), `rf_feature_control`(§4.5·§11), `a2_supervision`(§6 #7
콜아웃), `sr_labeling`(§4.4 #9 콜아웃) — 결론 유지(또는 confounded 표시), 귀속 교정.

---

## 5. 남은 점검 과제 (본류와 독립)

| # | 과제 | 동기 | 상태 |
|---|------|------|------|
| #7 | A2 관리종목 coverage 점검 | 11.8% vs 코스닥 상폐 80%+ 괴리 → 키워드 누락 의심 | **완료** — 실제 coverage 100%, A2 결론 confounded |
| #9 | v3(L7) valid 재평가 | SR §4.4 "valid 0.42는 착시" 가설 루프 닫기 | **완료** — 착시 재현(+0.166) |
| — | A2 재수집·재평가 (RF+딥) | #7 후속: 확장 키워드 + 시점 컷오프로 관리종목 피처 재구성 | **완료** — 약한 RF엔 ROC/top-K 이득이나 **best 딥모델(T-38)엔 약손해(PR 0.458→0.426)**. A2 "운영 모델 불필요" 정정된 이유로 유지. 말기는 누수 (`research_a2_reeval_report.md`) |
| — | 시장 피처 재설계 | frozen 분리 후 *살아있는 거래주 추세*만 추출해 재검증 | 미착수 |
| — | 2025 데이터 OoS | 모든 통계가 in-sample(test 2024, 양성 56). 진짜 일반화 검증 | 데이터 대기 |

---

## 6. 재현성

```powershell
.venv\Scripts\python.exe -m src.research.rf_feature_control.run_rf_control       # #1
.venv\Scripts\python.exe -m src.research.rf_feature_control.run_wf_control       # #3
.venv\Scripts\python.exe -m src.research.rf_feature_control.run_audit_check      # #2·#4
.venv\Scripts\python.exe -m src.research.rf_feature_control.run_market_check     # #8
.venv\Scripts\python.exe -m src.research.rf_feature_control.run_frozen_check     # #8b
```

모든 seed·bootstrap 고정. 동일 seed 재실행 시 결과 일치. 추가 의존성 없음
(기존 S0/S1 환경 재사용).
