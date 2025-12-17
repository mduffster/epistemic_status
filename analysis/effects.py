"""
Effect size and ROC/AUC analysis.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve

from .loader import ModelData, get_activation_matrix


def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Compute Cohen's d effect size between two groups.

    Args:
        group1: Array of values for group 1
        group2: Array of values for group 2

    Returns:
        Cohen's d value
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (group1.mean() - group2.mean()) / pooled_std


def compute_effect_sizes(
    data: ModelData,
    position: str = 'last',
    print_output: bool = True
) -> Dict:
    """
    Compute effect sizes for activation differences between correct/incorrect.

    Returns:
        Dictionary with effect sizes by layer and aggregated
    """
    if print_output:
        print("\n" + "=" * 60)
        print("EFFECT SIZE ANALYSIS")
        print("=" * 60)

    df = data.df
    correct_mask = df['correct'].values.astype(bool)

    X = data.activations[f'resid_{position}']
    n_layers = X.shape[1]

    results = {
        'by_layer': {},
        'entropy': {},
        'summary': {}
    }

    # Effect size for entropy
    entropy_correct = df[correct_mask]['entropy'].values
    entropy_incorrect = df[~correct_mask]['entropy'].values
    d_entropy = compute_cohens_d(entropy_incorrect, entropy_correct)

    results['entropy'] = {
        'cohens_d': d_entropy,
        'interpretation': _interpret_cohens_d(d_entropy)
    }

    if print_output:
        print(f"\n--- Entropy Effect Size ---")
        print(f"Cohen's d: {d_entropy:.3f} ({_interpret_cohens_d(d_entropy)})")
        print("(Positive = incorrect has higher entropy)")

    # Effect size by layer (mean activation magnitude)
    layer_effects = []

    if print_output:
        print(f"\n--- Activation Effect Sizes by Layer ---")

    for layer in range(n_layers):
        X_layer = X[:, layer, :]

        # Mean activation magnitude per sample
        mag_correct = np.linalg.norm(X_layer[correct_mask], axis=1)
        mag_incorrect = np.linalg.norm(X_layer[~correct_mask], axis=1)

        d_mag = compute_cohens_d(mag_correct, mag_incorrect)
        layer_effects.append(d_mag)

        results['by_layer'][layer] = {
            'cohens_d_magnitude': d_mag
        }

        if print_output and layer % 7 == 0:  # Print every 7th layer
            print(f"Layer {layer:2d}: d={d_mag:.3f} ({_interpret_cohens_d(d_mag)})")

    # Summary statistics
    results['summary'] = {
        'mean_effect': np.mean(np.abs(layer_effects)),
        'max_effect': np.max(np.abs(layer_effects)),
        'max_effect_layer': int(np.argmax(np.abs(layer_effects))),
        'min_effect': np.min(np.abs(layer_effects)),
        'min_effect_layer': int(np.argmin(np.abs(layer_effects)))
    }

    if print_output:
        print(f"\n--- Summary ---")
        print(f"Mean |d| across layers: {results['summary']['mean_effect']:.3f}")
        print(f"Max |d|: {results['summary']['max_effect']:.3f} (layer {results['summary']['max_effect_layer']})")

    return results


def compute_roc_auc(
    data: ModelData,
    position: str = 'last',
    layer: Optional[int] = None,
    print_output: bool = True
) -> Dict:
    """
    Compute ROC curves and AUC for correctness prediction.

    Returns:
        Dictionary with ROC/AUC results for different predictors
    """
    if print_output:
        print("\n" + "=" * 60)
        print("ROC/AUC ANALYSIS")
        print("=" * 60)

    y = data.df['correct'].values.astype(int)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {}

    # 1. Entropy-based prediction
    X_entropy = data.df['entropy'].values.reshape(-1, 1)
    scaler_e = StandardScaler()
    X_entropy_scaled = scaler_e.fit_transform(X_entropy)

    clf_entropy = LogisticRegression(max_iter=1000, random_state=42)
    # Use cross_val_predict to get out-of-fold predictions
    probs_entropy = cross_val_predict(clf_entropy, X_entropy_scaled, y, cv=cv, method='predict_proba')[:, 1]
    auc_entropy = roc_auc_score(y, probs_entropy)
    fpr_e, tpr_e, _ = roc_curve(y, probs_entropy)

    results['entropy'] = {
        'auc': auc_entropy,
        'fpr': fpr_e,
        'tpr': tpr_e
    }

    if print_output:
        print(f"\n--- Entropy Only ---")
        print(f"AUC: {auc_entropy:.3f}")

    # 2. Full activation probe
    X_full = get_activation_matrix(data, position, layer)
    scaler_f = StandardScaler()
    X_full_scaled = scaler_f.fit_transform(X_full)

    clf_full = LogisticRegression(max_iter=1000, random_state=42)
    probs_full = cross_val_predict(clf_full, X_full_scaled, y, cv=cv, method='predict_proba')[:, 1]
    auc_full = roc_auc_score(y, probs_full)
    fpr_f, tpr_f, _ = roc_curve(y, probs_full)

    layer_desc = f"layer {layer}" if layer is not None else "all layers"
    results['probe'] = {
        'auc': auc_full,
        'fpr': fpr_f,
        'tpr': tpr_f,
        'layer': layer
    }

    if print_output:
        print(f"\n--- Activation Probe ({layer_desc}) ---")
        print(f"AUC: {auc_full:.3f}")

    # 3. Best single layer (if not already specified)
    if layer is None:
        X_resid = data.activations[f'resid_{position}']
        n_layers = X_resid.shape[1]
        best_auc = 0
        best_layer = 0

        for l in range(n_layers):
            X_layer = X_resid[:, l, :]
            scaler_l = StandardScaler()
            X_layer_scaled = scaler_l.fit_transform(X_layer)
            clf_l = LogisticRegression(max_iter=1000, random_state=42)
            probs_l = cross_val_predict(clf_l, X_layer_scaled, y, cv=cv, method='predict_proba')[:, 1]
            auc_l = roc_auc_score(y, probs_l)
            if auc_l > best_auc:
                best_auc = auc_l
                best_layer = l

        # Get full ROC for best layer
        X_best = X_resid[:, best_layer, :]
        scaler_b = StandardScaler()
        X_best_scaled = scaler_b.fit_transform(X_best)
        clf_best = LogisticRegression(max_iter=1000, random_state=42)
        probs_best = cross_val_predict(clf_best, X_best_scaled, y, cv=cv, method='predict_proba')[:, 1]
        fpr_b, tpr_b, _ = roc_curve(y, probs_best)

        results['best_layer'] = {
            'layer': best_layer,
            'auc': best_auc,
            'fpr': fpr_b,
            'tpr': tpr_b
        }

        if print_output:
            print(f"\n--- Best Single Layer (layer {best_layer}) ---")
            print(f"AUC: {best_auc:.3f}")

    # Summary
    if print_output:
        print(f"\n--- Summary ---")
        print(f"Entropy AUC:     {auc_entropy:.3f}")
        print(f"Full Probe AUC:  {auc_full:.3f}")
        if 'best_layer' in results:
            print(f"Best Layer AUC:  {results['best_layer']['auc']:.3f} (layer {results['best_layer']['layer']})")
        print(f"\nAUC improvement over entropy: +{(auc_full - auc_entropy):.3f}")

    return results


def compute_layer_auc(
    data: ModelData,
    position: str = 'last',
    print_output: bool = True
) -> pd.DataFrame:
    """
    Compute AUC for each layer.

    Returns:
        DataFrame with layer-wise AUC values
    """
    if print_output:
        print("\n--- Layer-wise AUC ---")

    y = data.df['correct'].values.astype(int)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    X_resid = data.activations[f'resid_{position}']
    n_layers = X_resid.shape[1]

    results = []

    for layer in range(n_layers):
        X_layer = X_resid[:, layer, :]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_layer)

        clf = LogisticRegression(max_iter=1000, random_state=42)
        probs = cross_val_predict(clf, X_scaled, y, cv=cv, method='predict_proba')[:, 1]
        auc = roc_auc_score(y, probs)

        results.append({'layer': layer, 'auc': auc})

        if print_output:
            print(f"Layer {layer:2d}: AUC={auc:.3f}")

    return pd.DataFrame(results)


def _interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size."""
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"
