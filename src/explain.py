"""
explain.py
SHAP explainability layer for the XAI SME Threat Detection project.

Builds a SHAP TreeExplainer over a trained tree model and extracts, for any
single alert, the top-k features that drove the model's decision for the
predicted class. Also provides the within-class explanation-consistency
measure (RQ2).

Core indexing note: shap_values has shape (n_samples, n_features, n_classes).
For one alert we slice shap_values[sample_idx, :, predicted_class] to get the
per-feature attribution for the class the model actually chose.
"""

import numpy as np
import shap
from itertools import combinations


def build_explainer(model, X_sample):
    """Build a TreeExplainer and compute SHAP values for a sample of rows.

    Returns: (explainer, shap_values)
    shap_values shape observed in the study:
      IDS2025    -> (100, 79, 7)
      UNSW-NB15  -> (100, 42, 10)
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    return explainer, shap_values


def top_k_features(shap_values, sample_idx, predicted_class, feature_names, k=5):
    """Return the top-k (name, signed SHAP value) pairs for one alert.

    Ranking is by absolute SHAP value (magnitude of influence), but the
    returned values keep their sign so direction is visible.
    """
    sample_shap = shap_values[sample_idx, :, predicted_class]
    top_idx = np.argsort(np.abs(sample_shap))[::-1][:k]
    return [(feature_names[i], float(sample_shap[i])) for i in top_idx]


def top_k_indices(shap_values, sample_idx, predicted_class, k=5):
    """Return just the indices of the top-k features (used by RQ2 and RQ6)."""
    sample_shap = shap_values[sample_idx, :, predicted_class]
    return np.argsort(np.abs(sample_shap))[::-1][:k]


def explanation_consistency(model, shap_values, X_sample, class_names, k=5):
    """RQ2: mean pairwise top-k Jaccard overlap of explanations within each class.

    Higher overlap = more consistent explanations for that class. In the study,
    consistency tracked behavioural homogeneity (PortScan 0.772) rather than
    class frequency (Normal 0.298).

    Returns: dict {class_name: (mean_jaccard, n_alerts)}
    """
    by_class = {}
    n = len(X_sample)
    for i in range(n):
        pc = int(model.predict(X_sample.iloc[[i]])[0])
        idx = set(top_k_indices(shap_values, i, pc, k).tolist())
        by_class.setdefault(pc, []).append(idx)

    out = {}
    for pc, sets in by_class.items():
        if len(sets) < 2:
            out[class_names[pc]] = (None, len(sets))
            continue
        jacs = [len(a & b) / len(a | b) for a, b in combinations(sets, 2)]
        out[class_names[pc]] = (float(np.mean(jacs)), len(sets))
    return out
