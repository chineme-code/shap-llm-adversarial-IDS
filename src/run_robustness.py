"""
run_robustness.py
Reproduces the RQ6 adversarial robustness evaluation on IDS2025.

WHAT IT DOES
------------
1. Rebuilds the IDS2025 model and data (same seed 42 split).
2. Builds a SHAP TreeExplainer.
3. Runs a single-alert attack demo (verdict held, explanation changed).
4. Runs the budget sweep over 100 alerts and saves it to
   results/robustness_sweep_100.json.

WHY
---
This is the paper's central safety finding: it tests whether the analyst-facing
SHAP explanation can be manipulated by a small, prediction-preserving
perturbation. In the study 83-88% of explanations were disrupted even at the
smallest budget while the model's verdict never changed.

RUN (from project root, venv active):
    python src/run_robustness.py
"""

import shap

from preprocess import load_ids2025, clean_ids2025, encode_and_split_ids2025
from train import train_xgboost
from attack_explanations import feature_std_vector, show_attack_example, run_budget_sweep


def main():
    df = clean_ids2025(load_ids2025("data/IDS2025.xlsx"))
    X_train, X_test, y_train, y_test, le = encode_and_split_ids2025(df)
    model = train_xgboost(X_train, y_train, len(le.classes_))
    explainer = shap.TreeExplainer(model)

    stds = feature_std_vector(X_train)
    feature_names = list(X_test.columns)

    print("== Single-alert demo ==")
    show_attack_example(model, explainer, X_test.iloc[[0]], stds,
                        feature_names, le.classes_, budget=0.5, n_trials=500, seed=1)

    print("\n== Budget sweep (100 alerts, 200 trials/alert) ==")
    run_budget_sweep(
        model, explainer, X_test.iloc[:100], stds, feature_names,
        budgets=(0.05, 0.1, 0.25, 0.5, 1.0), n_trials=200,
        save_path="results/robustness_sweep_100.json",
    )


if __name__ == "__main__":
    main()
