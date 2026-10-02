# Results

This file records the actual outputs produced by the scripts in this repository,
organized by research question (RQ). All numbers are the real values from the
experimental runs. Where a result is a screenshot in the original work, the exact
figures are transcribed here.

Datasets:
- **IDS2025**: refined, class-rebalanced CICIDS2017 derivative. 91,830 rows, 80 columns, 7 classes.
- **UNSW-NB15**: independent benchmark. 175,341 train / 82,332 test, 42 features, 10 classes.

---

## Dataset preparation (IDS2025)

| Item | Value |
|---|---|
| Rows x columns | 91,830 x 80 |
| Label column | `newLabel` |
| Nulls / Infinities removed | 77 / 177 (negligible) |
| Duplicate rows | 86 (0.09%) |
| Feature dtypes | 55 int, 24 float, 1 categorical |
| Split (stratified, seed 42) | 73,362 train / 18,341 test |

Class distribution (full dataset):

| Class | Count |
|---|---|
| Normal | 26,185 |
| Dos/DDos | 26,066 |
| PortScan | 25,409 |
| Brute Force | 10,201 |
| Web Attack | 2,067 |
| Botnet ARES | 1,873 |
| Infiltration | 29 |

Label encoding: `Botnet ARES=0, Brute Force=1, Dos/DDos=2, Infiltration=3, Normal=4, PortScan=5, Web Attack=6`

---

## RQ1 - Detection performance and baselines (IDS2025)

| Model | Accuracy | Macro-F1 |
|---|---|---|
| XGBoost | 0.9958 | 0.9688 |
| Random Forest | 0.9953 | 0.9673 |
| Logistic Regression | 0.9421 | 0.8384 |

XGBoost is marginally ahead of Random Forest and clearly ahead of Logistic
Regression. The 13-point macro-F1 gap over Logistic Regression indicates the
task is not linearly separable. XGBoost was chosen as the primary model because
it matches the best baseline while supporting exact, efficient SHAP explanations.

---

## RQ4 - Per-class performance (IDS2025, XGBoost, N = 18,341)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Botnet ARES | 0.99 | 1.00 | 1.00 | 373 |
| Brute Force | 1.00 | 1.00 | 1.00 | 2,040 |
| Dos/DDos | 0.99 | 1.00 | 0.99 | 5,199 |
| Infiltration | 1.00 | 0.67 | 0.80 | 6 |
| Normal | 1.00 | 0.99 | 0.99 | 5,233 |
| PortScan | 1.00 | 1.00 | 1.00 | 5,077 |
| Web Attack | 1.00 | 1.00 | 1.00 | 413 |
| **Accuracy** | | | **1.00** | 18,341 |
| Macro avg | | | 0.97 | 18,341 |

Infiltration recall (0.67 on 6 test samples) is a small-sample artifact from its
extreme rarity (29 total samples), not an algorithmic failure.

---

## Leakage verification (IDS2025)

Three independent checks, all passed:

1. **Duplicates**: 86 rows (0.09%) - too few to inflate scores.
2. **Feature importance** (top 10, spread out and healthy - no single dominant feature):

| Feature | Importance |
|---|---|
| Min Packet Length | 0.289 |
| Bwd Packets/s | 0.166 |
| PSH Flag Count | 0.126 |
| Active Max | 0.073 |
| Destination Port | 0.063 |
| Bwd Packet Length Std | 0.050 |
| FIN Flag Count | 0.041 |
| Init_Win_bytes_backward | 0.035 |
| Total Fwd Packets | 0.018 |
| Bwd Packet Length Min | 0.015 |

3. **Destination Port ablation**: removing the one feature known in the
   literature to act as a label proxy changed almost nothing.

| Condition | Accuracy | Macro-F1 |
|---|---|---|
| With Destination Port | 0.9958 | 0.9688 |
| Without Destination Port | 0.9956 | 0.9681 |

---

## RQ2 - Explanation consistency (IDS2025, first 100 test alerts)

Mean pairwise top-5 Jaccard overlap within each class:

| Class | Mean Jaccard | n |
|---|---|---|
| PortScan | 0.772 | 31 |
| Brute Force | 0.624 | 9 |
| Web Attack | 0.508 | 3 |
| Dos/DDos | 0.392 | 28 |
| Normal | 0.298 | 27 |
| Botnet ARES | 0.250 | 2 |

Consistency tracks behavioural homogeneity, not class frequency: structured
PortScan traffic yields highly consistent explanations, while heterogeneous
benign Normal traffic yields the least consistent.

---

## RQ3 - False positives (IDS2025)

| Class | Support | FP | FN | FN-rate |
|---|---|---|---|---|
| Botnet ARES | 373 | 2 | 0 | 0.000 |
| Brute Force | 2,040 | 1 | 0 | 0.000 |
| Dos/DDos | 5,199 | 63 | 7 | 0.001 |
| Infiltration | 6 | 0 | 2 | 0.333 |
| Normal | 5,233 | 10 | 64 | 0.012 |
| PortScan | 5,077 | 1 | 3 | 0.001 |
| Web Attack | 413 | 0 | 1 | 0.002 |

**Normal traffic misclassified as an attack (false alarms): 64 of 5,233 = 1.22%.**
A low false-alarm rate is critical to prevent alert fatigue for a lean SME team.

---

## Explainability layer (SHAP + Claude)

- SHAP TreeExplainer output shape (IDS2025): **(100, 79, 7)** = samples x features x classes.
- Example top-5 for a Botnet ARES alert: Fwd Header Length (3.93), Source Port (2.78),
  Init_Win_bytes_forward (2.42), Init_Win_bytes_backward (2.37), Destination Port (0.43).
- The Claude translation layer produced correct, class-specific three-sentence
  summaries across Botnet ARES, Brute Force, PortScan, and Normal, and (on
  UNSW-NB15) Fuzzers, confirming it reasons about each classification rather than
  templating.

---

## Generalization - UNSW-NB15

| Model | Accuracy | Macro-F1 |
|---|---|---|
| XGBoost | 0.7660 | 0.5024 |
| Random Forest | 0.7543 | 0.4704 |
| Logistic Regression | 0.6720 | 0.3190 |

Per-class (XGBoost): Generic F1 0.98, Normal 0.85, Reconnaissance 0.87,
Exploits 0.71 detect reliably; rare classes collapse (Analysis 0.08, Backdoor 0.06,
DoS 0.18), each with under 700 test samples. Feature importance was spread out
(top feature `ct_dst_sport_ltm` 0.259); duplicates were 0 in both partitions.
SHAP shape: **(100, 42, 10)**.

The realistic drop from 99.58% to 76.60% shows the pipeline generalizes and is
learning genuine signal, and is indirect evidence against leakage on IDS2025.

---

## RQ6 - Adversarial robustness of explanations (IDS2025)

**Single-alert demo** (Botnet ARES, budget 0.5, seed 1): prediction UNCHANGED,
top-5 Jaccard overlap 0.667, Spearman 0.990. Three of the five cited features
changed while the verdict and its confidence were preserved.

**Budget sweep** (100 alerts, 200 trials/alert). Full data in
`robustness_sweep_100.json`:

| Budget (std) | Mean top-5 overlap | Min overlap | % disrupted | Mean Spearman |
|---|---|---|---|---|
| 0.05 | 0.315 | 0.000 | 88% | 0.920 |
| 0.10 | 0.316 | 0.000 | 88% | 0.917 |
| 0.25 | 0.332 | 0.000 | 85% | 0.914 |
| 0.50 | 0.336 | 0.000 | 83% | 0.907 |
| 1.00 | 0.333 | 0.000 | 84% | 0.905 |

**Finding.** SHAP explanations are fragile to prediction-preserving perturbation:
83-88% of alerts had their top-5 explanation meaningfully altered even at the
smallest budget (0.05 std), while the model's verdict never changed. Disruption
does not grow much with budget, so the fragility is present even for a minimal
attack. Notably, the full-ranking Spearman correlation stays high (~0.91): the
attack churns the top-5 set an analyst actually reads more than it scrambles the
whole ranking, which is precisely the part that matters for a top-k dashboard.

**Caveats.** Random search is a weak adversary, so these figures are a lower
bound on attackability. Not all perturbations are guaranteed to be physically
realizable network traffic. The sweep was run on IDS2025 only.

---

## 20-alert sweep (earlier, superseded by the 100-alert run above)

| Budget | Mean overlap | Min | % disrupted |
|---|---|---|---|
| 0.05 | 0.351 | 0.000 | 85% |
| 0.10 | 0.353 | 0.000 | 85% |
| 0.25 | 0.337 | 0.000 | 85% |
| 0.50 | 0.308 | 0.000 | 85% |
| 1.00 | 0.323 | 0.000 | 85% |

The 100-alert run confirmed the 20-alert pattern held, giving sturdier numbers.
