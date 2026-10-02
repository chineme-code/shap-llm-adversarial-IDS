# XAI SME Threat Detection

**Explainable intrusion detection for resource-constrained organizations: a SHAP-LLM framework with adversarial robustness evaluation.**

This repository contains the full, reproducible code for a study that builds an
explainable network intrusion detection pipeline aimed at small and medium
enterprises (SMEs) and then stress-tests whether its explanations can be trusted
under adversarial conditions.

The pipeline has three layers:

1. **Detect** - an XGBoost classifier flags network flows as benign or one of several attack types.
2. **Explain** - SHAP (TreeExplainer) identifies which features drove each individual decision.
3. **Translate** - the Claude API turns that technical attribution into a short, plain-language summary a non-expert analyst can act on.

It is evaluated on two datasets (IDS2025 and UNSW-NB15) and includes an
adversarial robustness test (RQ6) showing that the SHAP explanations are fragile
to small, prediction-preserving perturbations.

> This is a research and portfolio project. It is a prototype that validates the
> approach on benchmark data, not a production security product. See Limitations.

---

## Why this project exists

SMEs face enterprise-scale threats but usually have no Security Operations Center,
just one overworked IT generalist. Most ML security tools output a bare verdict
("malicious, 94%") with no reason, which a non-expert cannot act on. This project
closes that gap by translating the model's reasoning into plain language, and then
asks the question almost no prior work asks: **can that reason be manipulated?**

---

## What each file does, and why

### `src/` - the pipeline modules

| File | What it does | Why it matters |
|---|---|---|
| `preprocess.py` | Loads, cleans, encodes, and splits both datasets. Handles IDS2025 (all-numeric) and UNSW-NB15 (three categorical columns, name/content-swapped partitions). | Reproducible data prep is the foundation; the fixed seed (42) makes every downstream number repeatable. |
| `train.py` | Trains XGBoost and the Random Forest / Logistic Regression baselines, prints the per-class report, and runs the Destination Port leakage ablation. | Establishes detection performance (RQ1/RQ4) and proves the accuracy is genuine, not leakage. |
| `explain.py` | Builds the SHAP TreeExplainer, extracts the top-k features per alert, and computes within-class explanation consistency (RQ2). | This is the explainer layer, the core of the "why did it fire" answer. |
| `alert_generator.py` | Sends the SHAP top features to the Claude API and returns a constrained three-sentence analyst summary. | This is the translation layer that makes the output usable by a non-expert. |
| `attack_explanations.py` | The RQ6 adversarial attack: a prediction-preserving random-search perturbation that tries to change the SHAP explanation while keeping the verdict fixed. | The central research contribution: it tests whether the explanation can be trusted under attack. |

### `src/` - the runnable end-to-end scripts

| File | What it does |
|---|---|
| `run_ids2025.py` | Reproduces all IDS2025 results (RQ1-RQ4 + leakage checks) and saves the model + dashboard artifacts. |
| `run_unsw.py` | Reproduces the UNSW-NB15 generalization test. |
| `run_robustness.py` | Reproduces the RQ6 single-alert demo and the 100-alert budget sweep. |

### Other directories

| Path | What it holds |
|---|---|
| `notebooks/01_full_pipeline.ipynb` | The whole study as an annotated notebook, with a short brief before every cell. |
| `dashboard/app.py` | The Streamlit "SOC Alert Explainability Dashboard" that shows classification + SHAP chart + plain-language summary, in live or pre-generated mode. |
| `results/RESULTS.md` | Every result, transcribed with the real numbers, organized by research question. |
| `results/robustness_sweep_100.json` | Raw RQ6 sweep output (100 alerts). |
| `results/robustness_sweep_20.json` | Earlier 20-alert sweep (superseded, kept for provenance). |
| `data/README.md` | Where to download the datasets (not committed). |

---

## Research questions answered

| RQ | Question | Headline result |
|---|---|---|
| RQ1 | Detection performance vs baselines | XGBoost 99.58% acc, 0.9688 macro-F1 (beats RF and LR) |
| RQ2 | Are explanations consistent within a class? | Class-dependent: PortScan 0.772, Normal 0.298 |
| RQ3 | Can it suppress false alarms? | Yes: 1.22% of benign flows misclassified |
| RQ4 | Per-class limits | Strong except Infiltration (6 test samples) |
| RQ5 | Cost-benefit for SMEs | Near-zero open-source deployment (see RESULTS.md / paper) |
| RQ6 | Are explanations robust to attack? | No: 83-88% disrupted at minimal budget, verdict preserved |
| Gen. | Does it generalize to another dataset? | Yes: UNSW-NB15 76.60% acc, realistic decline |

Full numbers and tables are in [`results/RESULTS.md`](results/RESULTS.md).

---

## Quick start

```bash
# 1. Clone and enter
git clone https://github.com/chineme-code/shap-llm-adversarial-IDS.git
cd shap-llm-adversarial-IDS

# 2. Create and activate a virtual environment
python3 -m venv xai-sme-env
source xai-sme-env/bin/activate      # Windows: xai-sme-env\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Claude API key (only needed for the translation layer)
cp .env.example .env
#   then edit .env and paste your real key after ANTHROPIC_API_KEY=

# 5. Download the datasets into data/  (see data/README.md)

# 6. Reproduce the results
python src/run_ids2025.py       # detection + leakage checks, saves the model
python src/run_unsw.py          # generalization test
python src/run_robustness.py    # RQ6 adversarial robustness

# 7. Launch the dashboard (from the project root)
streamlit run dashboard/app.py
```

Or open `notebooks/01_full_pipeline.ipynb` to run the whole study interactively
with an explanation before every step.

---

## Reproducibility notes

- The 80/20 split uses `random_state=42`, so the exact train/test partition
  (73,362 / 18,341 for IDS2025) reproduces on any machine.
- The RQ6 attack seeds each alert's search by its index, so the sweep is
  deterministic run to run.
- The datasets are not committed; download them per `data/README.md`.
- The API key is never committed (`.gitignore` excludes `.env`).

---

## Limitations

- Benchmark datasets overstate live-traffic performance; a feature-extraction
  layer would be needed to map real firewall or cloud flow logs to the trained
  feature schema.
- The RQ6 attack uses random search over an unconstrained feature space, so its
  numbers are a lower bound on attackability, and not every perturbation is
  guaranteed to be a physically realizable packet.
- IDS2025 does not yet have a peer-reviewed validation paper.
- No formal user study of analyst cognitive load has been run.

---

## Citation

If you use this code, please cite the associated prior framework:

> Osholake, S. F., Umealajekwu, C., Edohen, A., Majekodunmi, A. O., & Evans-Anoruo, U. (2024). Human-AI Collaborative Security Operations: Optimizing SOC Analyst Cognitive Load Through Augmented Intelligence Frameworks. Iconic Research and Engineering Journals, 8(6).

## License

MIT - see [LICENSE](LICENSE).

## Author

Chinemelum Umealajekwu - GitHub [@chineme-code](https://github.com/chineme-code)
