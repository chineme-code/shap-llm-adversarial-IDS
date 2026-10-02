"""
attack_explanations.py
Adversarial robustness testing for the XAI SME Threat Detection pipeline (RQ6).

Tests whether a small, PREDICTION-PRESERVING perturbation to a network flow can
change which features SHAP attributes the decision to. If the model's verdict is
unchanged but the top-k explanation changes, the analyst receives a correct
classification paired with a manipulated rationale.

Method: black-box random search (XGBoost + SHAP TreeExplainer are not
differentiable in the way gradient attacks need). The per-feature perturbation
budget is scaled by each feature's standard deviation so perturbations are
proportionate across features on very different scales.

Study result (IDS2025, 100 alerts, 200 trials/alert): 83-88% of explanations
disrupted (top-5 Jaccard < 0.6) even at the smallest budget (0.05 std), while
full-ranking Spearman correlation stayed high (~0.91).
"""

import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def feature_std_vector(X_train):
    """Per-feature standard deviations used to scale perturbations.
    Constant (zero-std) features get 0 so they are never perturbed.
    """
    stds = X_train.std().values.astype(float)
    stds[stds == 0] = 0.0
    return stds


def _predicted_class(model, x_df):
    return int(model.predict(x_df)[0])


def _top_k_indices(shap_row, k):
    return np.argsort(np.abs(shap_row))[::-1][:k]


def attack_sample(model, explainer, x_row_df, feature_stds, feature_names,
                  budget=0.1, n_trials=200, top_k=5, seed=0):
    """Search for the strongest prediction-preserving explanation attack on one alert.

    Returns a dict with the original prediction, original/attacked top-k feature
    indices, the best (lowest) Jaccard overlap achieved, the Spearman correlation
    of the full ranking at that point, and the adversarial feature vector.
    """
    rng = np.random.default_rng(seed)
    x = x_row_df.values.astype(float).flatten()

    orig_pred = _predicted_class(model, x_row_df)
    orig_shap = explainer.shap_values(x_row_df)[0, :, orig_pred]
    orig_order = _top_k_indices(orig_shap, top_k)
    orig_top = set(orig_order.tolist())

    best = {
        "orig_pred": orig_pred,
        "orig_order": orig_order,
        "best_order": orig_order,
        "best_jaccard": 1.0,
        "best_spearman": 1.0,
        "x_adv": x.copy(),
        "n_valid": 0,
    }

    for _ in range(n_trials):
        delta = rng.uniform(-budget, budget, size=x.shape) * feature_stds
        x_adv = x + delta
        x_adv_df = pd.DataFrame([x_adv], columns=feature_names)

        # Constraint: the prediction must not change
        if _predicted_class(model, x_adv_df) != orig_pred:
            continue
        best["n_valid"] += 1

        new_shap = explainer.shap_values(x_adv_df)[0, :, orig_pred]
        new_order = _top_k_indices(new_shap, top_k)
        new_top = set(new_order.tolist())

        jac = len(orig_top & new_top) / len(orig_top | new_top)
        if jac < best["best_jaccard"]:
            rho = spearmanr(np.abs(orig_shap), np.abs(new_shap)).correlation
            best["best_jaccard"] = jac
            best["best_spearman"] = float(rho) if rho == rho else 0.0  # guard NaN
            best["best_order"] = new_order
            best["x_adv"] = x_adv

    return best


def show_attack_example(model, explainer, x_row_df, feature_stds, feature_names,
                        class_names, budget=0.5, n_trials=500, top_k=5, seed=0):
    """Run the attack on one alert and print original vs. attacked top-k side by side."""
    r = attack_sample(model, explainer, x_row_df, feature_stds, feature_names,
                      budget=budget, n_trials=n_trials, top_k=top_k, seed=seed)

    orig_feats = [feature_names[i] for i in r["orig_order"]]
    adv_feats = [feature_names[i] for i in r["best_order"]]

    print(f"Prediction: {class_names[r['orig_pred']]}  (UNCHANGED after attack)")
    print(f"Budget: {budget} std   Valid perturbations found: {r['n_valid']}/{n_trials}")
    print(f"Top-{top_k} overlap (Jaccard): {r['best_jaccard']:.3f}   "
          f"Rank corr (Spearman): {r['best_spearman']:.3f}")
    print()
    print(f"{'ORIGINAL top-' + str(top_k):<40}{'AFTER ATTACK top-' + str(top_k)}")
    print("-" * 78)
    for a, b in zip(orig_feats, adv_feats):
        marker = "  " if a == b else " *"
        print(f"{a:<40}{b}{marker}")
    print()
    print("* = feature the analyst would now see cited that they did not before")
    return r


def run_budget_sweep(model, explainer, X_samples, feature_stds, feature_names,
                     budgets=(0.05, 0.1, 0.25, 0.5, 1.0), n_trials=200, top_k=5,
                     disrupt_threshold=0.6, save_path=None):
    """Run the attack across a range of budgets over a set of alerts.

    Produces the RQ6 threshold table. An alert counts as 'disrupted' if its best
    top-k overlap falls below disrupt_threshold at that budget.

    Returns a list of per-budget result dicts and (optionally) saves them as JSON.
    """
    results = []
    for budget in budgets:
        jaccards, spearmans = [], []
        for i in range(len(X_samples)):
            row = X_samples.iloc[[i]]
            r = attack_sample(model, explainer, row, feature_stds, feature_names,
                              budget=budget, n_trials=n_trials, top_k=top_k, seed=i)
            jaccards.append(r["best_jaccard"])
            spearmans.append(r["best_spearman"])

        jaccards = np.array(jaccards)
        entry = {
            "budget": float(budget),
            "mean_jaccard": float(jaccards.mean()),
            "min_jaccard": float(jaccards.min()),
            "mean_spearman": float(np.mean(spearmans)),
            "pct_disrupted": float((jaccards < disrupt_threshold).mean() * 100),
            "n_samples": int(len(X_samples)),
        }
        results.append(entry)
        print(f"budget={budget:<5}  mean top-{top_k} overlap={entry['mean_jaccard']:.3f}  "
              f"min={entry['min_jaccard']:.3f}  {entry['pct_disrupted']:.0f}% disrupted")

    if save_path:
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved sweep to {save_path}")
    return results
