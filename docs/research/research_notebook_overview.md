# 연구노트 종합 정리 — S0 진단부터 인과 귀속 점검까지

## 한국 코스닥 상장폐지 조기경보: "0.2876 천장"의 추적 기록

작성일: 2026-06-20
작성자: 이현지
범위: `docs/research_s0_*` ~ `research_frozen_check_*` 전체 (S0 → SR → S1 → A → best_ci → 점검 스위트)

---

## 0. 연구 문제 정의

- **목표:** DART 재무제표 기반으로 한국 코스닥 기업의 **상장폐지를 1년 전 예측**.
- **데이터:** `processed_fixed_v1` 패널(1,536종목, 분기). train 2015–2022 / valid
  2023 / test 2024. 라벨 fixed_N1(상폐 정확히 1년 전 = 양성).
- **난점:** 극심한 불균형(약 124:1), test 2024 **양성 56개**. 주지표 = **PR-AUC**.
- **baseline:** Random Forest, **33 피처**(재무비율 27 + 거시 6; HIGH_MISSING YoY
  3종 제외), `signed_log1p` 스케일링. test PR-AUC = **0.2876**.
- **출발 의문:** exp_010~018(RF/XGB/LGBM Optuna, FE, class_weight, SMOTE,
  Piotroski, fixed_N2) 9개 개선 실험이 모두 0.2876을 못 넘었다. **0.2876은
  진짜 천장인가, 표본 추출 분포의 한 점인가?**

연구는 5개 국면(S0 → SR → S1 → A → best_ci)으로 진행됐고, 2026-06-20 **점검
스위트**가 그 결론의 인과 귀속을 재검증했다. 아래는 시간순 기록과 결론의 진화다.

---

## 1. 시간순 연구 기록

### S0 — Baseline 진단 (2026-05-28) · `research_s0_diagnostic_report.md`
- **질문:** 0.2876 천장은 얼마나 견고한가?
- **데이터/방법:** test 2024 + 6개 진단(A1 Bayesian/frequentist bootstrap, A2
  4-fold walk-forward × 5-seed, A3 baseline posterior 기반 paired + Holm 보정,
  A4 PSI/KL/JS + 다변량 MMD, A5 비지도 5종(IF/LOF/OCSVM/HBOS/ECOD), A6
  permutation/SHAP/ECE).
- **핵심 결과:**
  - PR-AUC 0.2876의 **95% CI [0.180, 0.421]** (폭 0.24), 5-seed 평균 0.270±0.015.
  - exp_010~018 **9개 전부 CI 안**, Holm 보정 후 유의차 0개.
  - 분포 이동은 **거시변수(연도 키)가 지배**, 재무비율은 안정(부채비율만 예외).
    valid(2023)→test(2024) MMD > train→valid → valid가 test 예고편으로 부적합.
  - supervised RF가 모든 비지도를 0.13~0.25 능가 → **라벨에 진짜 신호 있음**.
  - ECE 0.003(잘 보정). 남은 오차 = **35개 confident FN**(상폐 1년 전에도 재무상
    정상으로 보이는 기업).
- **결론:** 천장은 sharp하지 않고 넓다. 단일 스냅샷+RF 튜닝으론 못 깬다. →
  **S1 불규칙 시계열·S3 hazard 승격**, S4 분포적응 후순위.

### SR — 라벨링 점검 (2026-05-29) · `research_sr_labeling_report.md`
- **질문:** 라벨 정의를 바꾸면(확장하면) 천장을 깨나?
- **데이터/방법:** 7개 train 라벨링(L1 fixed_N1 ~ L7 v3 all-years), valid/test는
  항상 L1로 고정 평가. RF + walk-forward + bootstrap.
- **핵심 결과:** 7개 모두 [0.234, 0.263] (폭 0.029), 전부 S0 CI 안. **L3
  rolling_H24**가 가장 일관 우수(P@20 0.50). v3의 "valid 0.42 붕괴"는 라벨 문제가
  아닌 **평가 비일관성 착시**로 재해석.
- **결론:** 라벨링은 ±3% 영향뿐, 천장 못 깸. **default 라벨링 L3 채택**, 이후
  stream은 시계열/hazard로.

### S1 Phase 1 — GRU (2026-05-29) · `research_s1_phase1_report.md`
- **질문:** 같은 33피처를 K=3 시퀀스로 GRU에 넣으면?
- **방법:** 1-layer GRU(hidden 64) + focal, L3 라벨, 5-seed.
- **결과:** 5-seed 평균 **0.304±0.046** vs RF 0.269 (+0.035). ensemble 0.376
  [0.257,0.513]. **단 seed 분산 RF의 3배.**
- **결론:** 시계열 입력은 의미 있는 채널. 안정성이 다음 과제.

### S1 Phase 2 — GRU-D (2026-05-30) · `research_s1_phase2_report.md`
- **질문:** 결측을 명시 모델링(GRU-D)하면 더 나은가?
- **방법:** `combined_raw`(NaN 보존), feature-level decay. 5-seed.
- **결과:** **negative** — 0.267±0.068 < GRU 0.304. 추가 파라미터 ~2,242개가
  양성 641개에서 overfit. (ROC 0.875는 최고.)
- **결론:** 모델 복잡도↑보다 **시계열 표현 다양화**가 유망. *(주: Phase1과
  데이터 소스가 달라 confound 있음 — 점검 대상이었음.)*

### S1 Phase 3 — Transformer + Time2Vec (2026-05-30) · `research_s1_phase3_report.md`
- **질문:** self-attention + 시간 임베딩은?
- **방법:** Time2Vec + 1-layer encoder(dim 64, heads 4), L3, 5-seed.
- **결과:** 5-seed 평균 **0.342±0.023** (std GRU의 절반). P@20 **0.70**, ROC 0.889.
  → **새 best.**
- **결론:** 딥모델 중 가장 안정·정확. Transformer를 default로.

### S1 Walk-forward 재검증 (2026-05-30) · `research_s1_walkforward_report.md`
- **질문:** Transformer 우위가 단일 split(어려운 2023 valid) 우연인가?
- **방법:** 4-fold × 3-seed, GRU vs Transformer.
- **결과:** Transformer **4/4 fold 우위**, 평균 +0.076(0.289 vs GRU 0.213),
  std 60%. *(RF의 fold별 test는 미산출 — 후일 점검 #3에서 보완.)*
- **결론:** Transformer 우위는 robust. 외부 데이터 단계로.

### A3 — 시장 피처 (2026-05-30) · `research_a3_market_features_report.md`
- **방법:** FinanceDataReader OHLCV → 분기말 6개 시장 피처(39 features).
- **결과:** Transformer wf 0.289→**0.297** (+0.008). ROC +0.033이나 **P@20 하락**.
- **결론:** "시장은 ranking 보조일 뿐 천장 못 깸. 직접 신호(A1/A2) 필요."

### A1 — 감사의견 피처 (2026-05-31) · `research_a1_audit_features_report.md`
- **방법:** DART 감사의견 → 5개 피처(44 features). 시점누수 차단(Y-1 의견만).
- **결과:** Transformer wf 0.289→**0.408** (+0.119), 4/4 fold +0.10↑, fold4 0.444.
  P@20 0.575→0.650. 교차표: 비적정 2연속 → 양성 38.5%(96배).
- **결론:** **감사가 결정적**, 천장 돌파에 근접. → 새 best baseline(44 features).

### A2 — 관리종목 피처 (2026-05-31) · `research_a2_supervision_features_report.md`
- **방법:** KRX 거래소공시 → 6개 피처(50 features).
- **결과:** **negative** — 0.408→0.402 (A1과 중복). coverage 11.8%뿐.
- **결론:** A1이 A2 신호 대부분 포함. A2 미채택. *(coverage 괴리는 점검 #7 과제.)*

### best_ci — 천장 돌파 통계 입증 (2026-05-31) · `research_best_ci_report.md`
- **질문:** best 모델이 baseline CI를 통계적으로 넘는가?
- **방법:** 단일 split, Transformer-44 vs RF-33, **paired Bayesian bootstrap** n=5,000.
- **결과:** best 5-seed 평균 0.432, ensemble **0.454** [0.331,0.586]. **Δ +0.183
  [+0.094,+0.279], P(best>base)=1.000.**
- **결론(당시):** "단순 RF(0.27) → 시계열 Transformer+부실신호(0.45), 천장 깨짐."

---

## 2. 점검 스위트 (2026-06-20) — 인과 귀속 재검증

best_ci의 +0.183은 **모델·시계열·외부피처 3변화가 혼재**되어 무엇이 동력인지
분리되지 않았다. 6개 점검 실험이 이를 분리·검증했다. (상세: `research_inspection
_overview.md`)

| # | 핵심 질문 | 결과 |
|---|-----------|------|
| **#1** | 시계열 vs 피처, 무엇이 동력? (단일 split) | 스냅샷 RF+44 = 0.413 ≈ T-44; **Δ_ts_full +0.039 P=0.83 비유의** (시장 포함 조건) |
| **#3** | 검정력 4배(walk-forward)로도? | RF-44 wf **0.399 ≈ A1의 0.408**; Δ_ts_full +0.021 P=0.69 비유의 |
| **#2** | 감사는 자명한 트리거인가? | audit-only RF = **0.252** ≈ 재무 단독 → 단독 무효, 상호작용 신호 |
| **#4** | 감사는 누수인가? | 양성 56개 의견→상폐 **gap=2년 전수** → 누수 아님; 52%는 적정 의견인데 상폐 |
| **#8** | RF+market ROC 0.988의 정체? | 양성 **45%가 frozen(거래동결) 주식** (음성 1.3%) → "죽은 주식 탐지기" artifact |
| **#8b** | artifact 제거 후 시계열은? | 시장 제거(38feat) 시 **T−RF = +0.090, P=0.998 유의**; best = **T-38 = 0.458** |
| **#7** | A2 관리종목 coverage 11.8%? | 상폐기업 실제 coverage **65%→100%**(확장 키워드). **A2 negative result는 불완전 수집 confounded** |
| **#9** | v3 "valid 0.42" 정체? | valid 정답 L1→L7로 PR-AUC **0.193→0.359(+0.166)**, base 3.2배 → SR "착시" 가설 재현 |
| **모델군** | 외부데이터 이득 RF 특화? best 아키텍처? | 감사 이득 **모델 무관 일반화**(스냅샷 5종); 군 내부 무차별(RF≈LogReg, GRU≈TCN≈Transformer), **부스팅·LSTM만 열위**. 군 간 Transformer−LogReg은 단일 split 비유의→**walk-forward +0.060 유의**. 기여 **감사(+0.11)>시계열(+0.06)>알고리즘(≈0)** |

---

## 3. 결론의 진화 (연구노트 핵심)

| 시점 | 그 시점의 믿음 | 무엇이 바꿨나 |
|------|----------------|----------------|
| exp_010~018 | "0.2876은 천장이다" | — |
| **S0** | "천장이 아니라 넓은 noise band [0.18,0.42]" | bootstrap CI + Holm |
| **SR** | "라벨링으론 못 깬다" | 7 라벨링 전부 CI 안 |
| **S1** | "시계열이 천장을 올린다 (Transformer best)" | GRU/Transformer wf 우위 |
| **A1** | "감사 피처로 천장 돌파 (0.41)" | wf +0.119 |
| **best_ci** | "시계열+부실신호가 천장을 깼다 (Δ+0.18 유의)" | paired bootstrap |
| **#1·#3** | "돌파의 80%는 피처. 시계열 PR-AUC 순효과는 비유의" | RF에 같은 피처 부여 |
| **#2·#4** | "감사는 정당하나 단독 불충분 (양성 절반만)" | audit-only / leakage 표 |
| **#8** | "시장의 위력은 frozen artifact였다" | coverage/분포 진단 |
| **#8b (현재)** | **"시장 artifact를 빼면 시계열이 유의하게 이긴다. best = Transformer+감사(38feat), 시장 폐기"** | artifact 제거 후 paired |

---

## 4. 최종 종합

1. **0.2876은 천장이 아니었다** (S0): 넓은 noise band의 한 점. 단순 스냅샷 RF
   튜닝(exp_010~018)으론 못 움직인다.
2. **천장 돌파의 정당한 동력 = 감사 피처 + 시계열 모델링.**
   - 감사 피처: 누수 아님(gap 2년), 자명한 트리거 아님(단독 0.25), 양성 48% 포착.
   - 시계열: 시장 artifact를 걷어내면 스냅샷을 +0.090(P=0.998) 유의하게 능가.
3. **시장 피처는 함정이었다** (A3가 옳았다): 양성 45%가 frozen인 "죽은 주식"
   artifact라 RF를 허위로 부풀리고 Transformer엔 노이즈. **폐기 권장.**
4. **정정된 best 모델 = Transformer + 재무 + 감사 (38 features, 시장 제외),
   test PR-AUC 0.458** — 기존 44feat(0.454)보다 높고, OHLCV 수집 불필요.
5. **남은 진짜 천장 = 양성의 52%** (적정 감사의견 + 정상 거래 중인데 상폐).
   감사도 시장도 못 잡는 S0의 confident FN. 재무 궤적·hazard·2025 OoS의 영역.

### 방법론적 교훈 (포트폴리오 가치)
- **bootstrap CI 없이 단일 점추정 비교는 위험하다** (S0: ±0.06 SE가 실험 간
  차이와 동급).
- **혼재된 개선은 반드시 분해해야 한다** (best_ci +0.18 → 80%가 피처).
- **"강한 피처"는 누수·artifact를 의심하라** (감사는 통과, 시장은 frozen artifact).
- **negative/비유의 ≠ 무효** — 검정력(양성 56)과 교란(시장 artifact)을 통제하면
  결론이 뒤집힐 수 있다 (#8b).

---

## 5. 남은 과제

| 과제 | 동기 | 상태 |
|------|------|------|
| #7 A2 coverage 점검 | 11.8% vs 코스닥 상폐 80%+ | **완료** — 실제 100%, A2 confounded |
| #9 v3 valid 재평가 | SR "착시" 가설 루프 닫기 | **완료** — 착시 재현(+0.166) |
| A2 재수집·재평가 | #7 후속: 확장 키워드 + 시점 컷오프 + 딥모델 | **완료** — 약한 RF엔 ROC/top-K↑이나 best 딥모델엔 약손해; A2 "운영 불필요" 정정된 이유로 유지; 말기 누수 |
| A4 새 정보채널(주요사항보고서) | "정보>모델" 명제 확장 | **완료** — 자금조달·감자는 재무+감사와 중복(PR 0), audit만 additive (`research_a4_major_reports_report.md`) |
| 2025 OoS 검증 | unseen 연도 일반화 | **시도→보류** — 2026 라벨 미성숙(ROC 0.94 유지·PR 붕괴=censoring). 라벨 성숙(말~내년) 후 재실행 (`research_oos_2025_report.md`) |
| 시장 피처 재설계 | frozen 분리 후 *살아있는 거래주 추세*만 | 미착수 |
| confident FN(52%) 공략 | S3 hazard / 재무 궤적 심화 | 미착수 |

---

## 6. 문서·코드 지도

- **국면별 보고서:** `research_s0_diagnostic` / `research_sr_labeling` /
  `research_s1_phase1~3` / `research_s1_walkforward` / `research_a1~a3` /
  `research_best_ci`.
- **점검 스위트:** `research_inspection_overview`(개요) + `research_rf_feature
  _control` / `research_audit_check` / `research_market_check` /
  `research_frozen_check` / `research_supervision_v3_check` / `research_a2_reeval` /
  `research_model_family` / `research_a4_major_reports` / `research_oos_2025`.
- **점검 코드:** `src/research/rf_feature_control/` (run_rf_control, run_wf_control,
  run_audit_check, run_market_check, run_frozen_check, run_supervision_audit,
  run_v3_recheck, run_supervision_reeval, run_supervision_reeval_deep,
  run_snapshot_models, run_sequence_models, run_cross_family, run_cross_family_wf) +
  `src/research/a2_supervision/`
  (fetch_supervision_raw, supervision_features_v2) + `src/research/s1_irregular_ts/`
  (lstm_baseline, tcn).
- **본 문서:** 전체 arc의 최상위 인덱스 겸 연구노트.
