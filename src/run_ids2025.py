"""
run_ids2025.py
End-to-end reproduction of the IDS2025 results (RQ1-RQ4 + leakage checks).

WHAT IT DOES, IN ORDER
----------------------
1. Loads and cleans IDS2025 (drops nulls/infinities).
2. Label-encodes the target and makes the stratified 80/20 split (seed 42).
3. Trains XGBoost and prints the per-class classification report (RQ4).
4. Compares XGBoost vs Random Forest vs Logistic Regression (RQ1).
5. Runs the three leakage checks: duplicate count, feature importance,
   and the Destination Port ablation.
6. Saves the model and the dashboard artifacts (X_test sample, class names).

WHY
---
This is the detector + evidence-of-trust half of the study. It produces the
numbers reported in results/RESULTS.md and the .pkl the dashboard loads.

RUN (from project root, venv active):
    python src/run_ids2025.py
"""

import json
import pandas as pd
from sklearn.metrics import confusion_matrix

from preprocess import load_ids2025, clean_ids2025, encode_and_split_ids2025
from train import (train_xgboost, compare_baselines, per_class_report,
                   destination_port_ablation, save_model)


def main():
    # 1-2. Load, clean, encode, split
    df = clean_ids2025(load_ids2025("data/IDS2025.xlsx"))
    print("Shape after cleaning:", df.shape)
    print("Duplicate rows:", df.duplicated().sum())  # leakage check 1
    X_train, X_test, y_train, y_test, le = encode_and_split_ids2025(df)
    print("Train:", X_train.shape, "| Test:", X_test.shape)
    n_classes = len(le.classes_)

    # 3. Train + per-class report (RQ4)
    model = train_xgboost(X_train, y_train, n_classes)
    print("\n== Per-class report (RQ4) ==")
    y_pred = per_class_report(model, X_test, y_test, le.classes_)

    # 4. Baselines (RQ1)
    print("\n== Baseline comparison (RQ1) ==")
    compare_baselines(X_train, y_train, X_test, y_test, n_classes, xgb_pred=y_pred)

    # 5. Leakage checks 2 and 3
    print("\n== Feature importance (leakage check 2) ==")
    importances = pd.Series(model.feature_importances_, index=X_train.columns)
    print(importances.sort_values(ascending=False).head(10))
    print("\n== Destination Port ablation (leakage check 3) ==")
    destination_port_ablation(X_train, X_test, y_train, y_test, n_classes)

    # RQ3: Normal-as-attack false alarm rate
    cm = confusion_matrix(y_test, y_pred)
    ni = list(le.classes_).index("Normal")
    false_alarms = cm[ni].sum() - cm[ni, ni]
    print(f"\nRQ3 false alarms: {false_alarms} of {cm[ni].sum()} "
          f"({100*false_alarms/cm[ni].sum():.2f}%) Normal flows flagged as attack")

    # 6. Save model + dashboard artifacts
    save_model(model, "models/xgboost_ids2025.pkl")
    X_test.iloc[:50].to_csv("data/X_test_sample.csv", index=False)
    with open("models/class_names.json", "w") as f:
        json.dump(le.classes_.tolist(), f)
    print("\nSaved model and dashboard artifacts.")


if __name__ == "__main__":
    main()
