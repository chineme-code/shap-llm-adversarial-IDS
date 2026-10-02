"""
run_unsw.py
End-to-end reproduction of the UNSW-NB15 generalization test.

WHAT IT DOES
------------
1. Loads the UNSW-NB15 pre-split partitions (handling the name/content swap).
2. Label-encodes the three categorical feature columns (proto, service, state)
   and the target (attack_cat), and drops non-feature columns.
3. Trains XGBoost and prints the per-class report.
4. Compares XGBoost vs Random Forest vs Logistic Regression.
5. Runs the leakage checks (feature importance, duplicate counts).

WHY
---
UNSW-NB15 is a structurally different benchmark (42 features, 10 classes,
categorical columns IDS2025 lacked). A realistic, lower score here shows the
pipeline generalizes and is not memorising one easy dataset, which is also
indirect evidence against leakage on IDS2025.

RUN (from project root, venv active):
    python src/run_unsw.py
"""

import pandas as pd

from preprocess import load_unsw, encode_unsw
from train import train_xgboost, compare_baselines, per_class_report


def main():
    # 1-2. Load + encode
    train_df, test_df = load_unsw()
    print("Train:", train_df.shape, "| Test:", test_df.shape)
    X_train, y_train, X_test, y_test, le = encode_unsw(train_df, test_df)
    print("Features:", X_train.shape[1], "| Classes:", list(le.classes_))
    n_classes = len(le.classes_)

    # 3. Train + per-class report
    model = train_xgboost(X_train, y_train, n_classes)
    print("\n== Per-class report ==")
    y_pred = per_class_report(model, X_test, y_test, le.classes_)

    # 4. Baselines
    print("\n== Baseline comparison ==")
    compare_baselines(X_train, y_train, X_test, y_test, n_classes, xgb_pred=y_pred)

    # 5. Leakage checks
    print("\n== Feature importance ==")
    importances = pd.Series(model.feature_importances_, index=X_train.columns)
    print(importances.sort_values(ascending=False).head(10))
    print("\nTrain duplicates:", train_df.duplicated().sum())
    print("Test duplicates:", test_df.duplicated().sum())


if __name__ == "__main__":
    main()
