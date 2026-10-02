"""
train.py
Model training and baseline comparison for the XAI SME Threat Detection project.

Trains the primary detector (XGBoost) and two baselines (Random Forest,
Logistic Regression) under identical splits, and reports accuracy and
macro-F1. Also provides the Destination Port leakage ablation used on IDS2025.
"""

import joblib
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report


def train_xgboost(X_train, y_train, num_class, random_state=42):
    """Train the primary XGBoost multi-class classifier."""
    model = XGBClassifier(
        objective="multi:softmax",
        num_class=num_class,
        eval_metric="mlogloss",
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def compare_baselines(X_train, y_train, X_test, y_test, num_class,
                      xgb_pred=None, random_state=42):
    """Train XGBoost, Random Forest, and Logistic Regression and print
    accuracy + macro-F1 for each.

    Logistic Regression receives standardized features because it is
    sensitive to feature magnitude; the tree models do not, because they
    split on thresholds.

    Returns a dict of {model_name: (accuracy, macro_f1)}.
    """
    results = {}

    # XGBoost (primary)
    if xgb_pred is None:
        xgb = train_xgboost(X_train, y_train, num_class, random_state)
        xgb_pred = xgb.predict(X_test)
    results["XGBoost"] = (
        accuracy_score(y_test, xgb_pred),
        f1_score(y_test, xgb_pred, average="macro"),
    )

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    results["Random Forest"] = (
        accuracy_score(y_test, rf_pred),
        f1_score(y_test, rf_pred, average="macro"),
    )

    # Logistic Regression (scaled)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    lr = LogisticRegression(max_iter=1000, random_state=random_state)
    lr.fit(X_train_s, y_train)
    lr_pred = lr.predict(X_test_s)
    results["Logistic Regression"] = (
        accuracy_score(y_test, lr_pred),
        f1_score(y_test, lr_pred, average="macro"),
    )

    for name, (acc, f1) in results.items():
        print(f"{name:20} accuracy={acc:.4f}  macro-F1={f1:.4f}")
    return results


def per_class_report(model, X_test, y_test, class_names):
    """Print the per-class precision/recall/F1 classification report."""
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=class_names))
    return y_pred


def destination_port_ablation(X_train, X_test, y_train, y_test, num_class,
                              port_col="Destination Port", random_state=42):
    """Leakage check: retrain XGBoost without Destination Port and compare.

    In the study this changed accuracy by <0.03% (0.9958 -> 0.9956),
    confirming the model does not rely on port-as-label-proxy leakage.
    """
    Xtr = X_train.drop(columns=[port_col])
    Xte = X_test.drop(columns=[port_col])
    m = train_xgboost(Xtr, y_train, num_class, random_state)
    pred = m.predict(Xte)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="macro")
    print(f"WITHOUT {port_col}: accuracy={acc:.4f}  macro-F1={f1:.4f}")
    return acc, f1


def save_model(model, path):
    """Persist a trained model to disk."""
    joblib.dump(model, path)
    return path
