# Statistical Ceiling-Breaking in Korean Stock-Delisting Early Warning

**Hyeonji Lee** · Kwangwoon University, Dept. of Computer & Information Engineering
**Email**: wendydeer@naver.com · **Target**: UNIST Data Analytics Lab (Prof. Sungil Kim)

---

## Abstract

Prior Korean stock-delisting early-warning models on quarterly financial-ratio
snapshots had stalled at **PR-AUC ≈ 0.29** through nine reported improvement
attempts (RF/XGB/LGBM tuning, SMOTE, feature engineering, alternative labels),
with the source of the stagnation undiagnosed. This study applies a
diagnostic-first methodology to the problem: **(i)** Bayesian bootstrap
(Rubin 1981) for credible-interval estimation, **(ii)** walk-forward
cross-validation with multi-seed measurement, **(iii)** paired bootstrap
statistical testing of all prior experiments, **(iv)** PSI/KL/MMD distribution-
shift quantification, and **(v)** systematic external-data integration. The
diagnostic phase finds that the 0.2876 figure sits inside a wide 95% Bayesian
credible interval **[0.180, 0.421]**, and **all nine prior experiments are
statistically indistinguishable** from baseline under Holm-Bonferroni correction.
Building on this base, an irregular-time-series model (**Transformer + Time2Vec**,
K=3) combined with **external audit-opinion features** (DART OpenAPI, 16,896
calls, 5 features encoding KOSDAQ Listing Regulation §38 triggers) lifts the
model to PR-AUC **0.432 ± 0.026** on the original single split (5 seeds).
A paired Bayesian bootstrap (n = 5,000 on shared Dirichlet weights) shows
**Δ 95% CI [+0.094, +0.279]** — strictly positive — and **P(best > baseline) =
1.000**, statistically establishing that the original ceiling has been broken.
The empirical finding replicates **Geiger & Raghunandan (2002)** on Korean
KOSDAQ: a non-clean audit opinion in two consecutive years yields a positive
rate **96× the base rate**.

## Key Contributions

- **Diagnostic methodology**: first to apply Bayesian bootstrap CI + paired
  statistical testing + multivariate-MMD shift detection + walk-forward CV as
  the *primary* diagnostic framework for the Korean delisting problem.
- **Negative-result rigor**: nine prior models in this lineage (PR-AUC
  0.19–0.29) statistically reinterpreted as inside-noise rather than failures.
- **Cross-cultural theory test**: Geiger & Raghunandan (2002) audit-opinion
  signal **strongly replicates** on Korean KOSDAQ (96× lift). Campbell,
  Hilscher & Szilagyi (2008) market signal **only partially replicates**
  (P@20 actually *decreases* when added alone).
- **Ceiling broken**: PR-AUC 0.270 → 0.432 (+60% relative), paired Δ CI
  strictly positive at 95% confidence.

## Method Summary

33 quarterly financial-ratio features (DART) on **1,536 firms**, train
2015–2022 / valid 2023 / test 2024; `fixed-N1` labels (label = 1 iff
delisted exactly one year ahead); **124 : 1** class imbalance, 56 test
positives. Four architectures compared (GRU → GRU-D → Transformer + Time2Vec
→ planned Neural CDE) under focal loss (γ = 2), focal–prior coupling for
imbalance, 1-layer Transformer (model_dim = 64), 5-seed multi-run. External
data added in three stages: 6 market features (FinanceDataReader OHLCV,
quarter-end), 5 audit features (DART `accnutAdtorNmNdAdtOpinion.json`,
Y − 1 only to prevent look-ahead), 6 supervision-designation features
(DART exchange disclosures). All evaluation uses walk-forward 4-fold × 3–5
seeds with Bayesian-bootstrap CI throughout.

## Key Results (single split, test 2024)

| Model | Test PR-AUC (5-seed) | 95% Bayesian CI | P@20 | ROC-AUC |
|---|---|---|---|---|
| RF baseline (S0 canonical) | 0.269 ± 0.013 | [0.167, 0.402] | 0.58 | 0.855 |
| Transformer 33 features (Phase 3) | 0.342 ± 0.023 | — | 0.58 | 0.889 |
| **Transformer + market + audit (44 features)** | **0.432 ± 0.026** | **[0.331, 0.586]** | **0.65+** | **0.919** |
| Paired Δ vs baseline | **+0.183** | **[+0.094, +0.279]** | — | — |

P(best > baseline) under paired bootstrap = **1.000** (5,000 / 5,000 resamples).

## Connection to Data Analytics Lab

| Lab signature direction | This work's matching component |
|---|---|
| **Bayesian-bootstrap monitoring statistics** | Bayesian bootstrap as primary CI method, throughout S0/best-CI; Rubin (1981) |
| **Irregular time-series classification** (Neural CDE, FlowPath) | K=3 quarterly mix (Q1/H1/Q3/ANNUAL) handled by Time2Vec + Transformer; Neural CDE (Kidger 2020) prepared as Phase 4 |
| **Industrial anomaly detection** | Delisting reframed and tested as anomaly detection; supervised RF beats all unsupervised baselines (IsolationForest/LOF/OCSVM/HBOS) by 0.13–0.25 PR-AUC, validating supervised framing |
| **Rigorous statistical reporting** | Paired bootstrap + Holm-Bonferroni + Mann–Whitney + permutation MMD for every comparison; all seeds and weights fixed for reproducibility |

## Reproducibility & Materials

Eight peer-style diagnostic reports under `docs/research_*_report.md`
(S0 diagnostic, SR labeling, S1 Phase 1–3 + walk-forward, A1/A2/A3 external,
and best-CI). All runners under `src/research/{s0_diagnostic, sr_labeling,
s1_irregular_ts, a1_audit, a2_supervision, a3_market, best_ci}/`. Wall-time
≈ 5 hours CPU end-to-end; no GPU required for the reported results.

## Key References

Rubin (1981), *Bayesian Bootstrap*, *Ann. Stat.*; Geiger & Raghunandan (2002),
*Going-Concern Opinions*, *Auditing*; Shumway (2001), *Hazard Bankruptcy*,
*J. Bus.*; Campbell, Hilscher & Szilagyi (2008), *Distress Risk*, *JF*;
Cho et al. (2014), *GRU*, *EMNLP*; Vaswani et al. (2017), *Attention*,
*NeurIPS*; Che et al. (2018), *GRU-D*, *Sci. Rep.*; Kazemi et al. (2019),
*Time2Vec*; Kidger et al. (2020), *Neural CDE*, *NeurIPS*; KRX KOSDAQ Listing
Regulations §28–§38.
