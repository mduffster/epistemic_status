"""
Entropy analysis functions.
"""

from typing import Dict, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from .loader import ModelData, get_activation_matrix


def entropy_analysis(data: ModelData, print_output: bool = True) -> Dict:
    """
    Analyze entropy distributions by category and correctness.

    Returns:
        Dictionary with entropy statistics
    """
    if print_output:
        print("\n" + "=" * 60)
        print("ENTROPY ANALYSIS")
        print("=" * 60)

    df = data.df
    results = {'by_category': {}, 'overall': {}}

    # Entropy by category
    if print_output:
        print("\n--- Entropy by Category ---")

    for cat in df['category'].unique():
        cat_data = df[df['category'] == cat]
        correct_entropy = cat_data[cat_data['correct']]['entropy'].mean()
        incorrect_entropy = cat_data[~cat_data['correct']]['entropy'].mean()

        results['by_category'][cat] = {
            'mean': cat_data['entropy'].mean(),
            'std': cat_data['entropy'].std(),
            'correct_mean': correct_entropy,
            'incorrect_mean': incorrect_entropy
        }

        if print_output:
            print(f"{cat}:")
            print(f"  Mean: {cat_data['entropy'].mean():.3f}, Std: {cat_data['entropy'].std():.3f}")
            print(f"  Correct mean: {correct_entropy:.3f}")
            print(f"  Incorrect mean: {incorrect_entropy:.3f}")

    # Overall
    correct_entropy = df[df['correct']]['entropy'].mean()
    incorrect_entropy = df[~df['correct']]['entropy'].mean()

    results['overall'] = {
        'correct_entropy': correct_entropy,
        'incorrect_entropy': incorrect_entropy,
        'correlation': df['entropy'].corr(df['correct'])
    }

    if print_output:
        print("\n--- Entropy vs Correctness (overall) ---")
        print(f"Correct answers mean entropy: {correct_entropy:.3f}")
        print(f"Incorrect answers mean entropy: {incorrect_entropy:.3f}")

    return results


def entropy_vs_probe(
    data: ModelData,
    position: str = 'last',
    print_output: bool = True
) -> Dict:
    """
    Compare entropy-only prediction to full activation probe.

    Returns:
        Dictionary comparing entropy-only, single layer, and full probe
    """
    if print_output:
        print("\n" + "=" * 60)
        print("ENTROPY VS PROBE COMPARISON")
        print("=" * 60)

    y = data.df['correct'].values.astype(int)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {}

    # 1. Entropy alone
    X_entropy = data.df['entropy'].values.reshape(-1, 1)
    scaler_e = StandardScaler()
    X_entropy_scaled = scaler_e.fit_transform(X_entropy)

    clf_entropy = LogisticRegression(max_iter=1000, random_state=42)
    scores_entropy = cross_val_score(clf_entropy, X_entropy_scaled, y, cv=cv, scoring='accuracy')

    # Fit once for AUC
    clf_entropy.fit(X_entropy_scaled, y)
    probs_entropy = clf_entropy.predict_proba(X_entropy_scaled)[:, 1]
    auc_entropy = roc_auc_score(y, probs_entropy)

    results['entropy_only'] = {
        'accuracy': scores_entropy.mean(),
        'accuracy_std': scores_entropy.std(),
        'auc': auc_entropy
    }

    if print_output:
        print(f"\n--- Entropy Only ---")
        print(f"Accuracy: {scores_entropy.mean():.3f} (+/- {scores_entropy.std():.3f})")
        print(f"AUC: {auc_entropy:.3f}")

    # 2. Full activation probe
    X_full = get_activation_matrix(data, position)
    scaler_f = StandardScaler()
    X_full_scaled = scaler_f.fit_transform(X_full)

    clf_full = LogisticRegression(max_iter=1000, random_state=42)
    scores_full = cross_val_score(clf_full, X_full_scaled, y, cv=cv, scoring='accuracy')

    results['full_probe'] = {
        'accuracy': scores_full.mean(),
        'accuracy_std': scores_full.std()
    }

    if print_output:
        print(f"\n--- Full Activation Probe (all layers) ---")
        print(f"Accuracy: {scores_full.mean():.3f} (+/- {scores_full.std():.3f})")

    # 3. Find best single layer
    n_layers = data.n_layers
    best_layer_acc = 0
    best_layer = 0

    X_resid = data.activations[f'resid_{position}']

    for layer in range(n_layers):
        X_layer = X_resid[:, layer, :]
        scaler_l = StandardScaler()
        X_layer_scaled = scaler_l.fit_transform(X_layer)
        clf_layer = LogisticRegression(max_iter=1000, random_state=42)
        scores_layer = cross_val_score(clf_layer, X_layer_scaled, y, cv=cv, scoring='accuracy')
        if scores_layer.mean() > best_layer_acc:
            best_layer_acc = scores_layer.mean()
            best_layer = layer

    results['best_layer'] = {
        'layer': best_layer,
        'accuracy': best_layer_acc
    }

    if print_output:
        print(f"\n--- Best Single Layer Probe (layer {best_layer}) ---")
        print(f"Accuracy: {best_layer_acc:.3f}")

    # 4. Entropy + best layer combined
    X_layer_best = X_resid[:, best_layer, :]
    X_combined = np.hstack([X_entropy, X_layer_best])
    scaler_c = StandardScaler()
    X_combined_scaled = scaler_c.fit_transform(X_combined)

    clf_combined = LogisticRegression(max_iter=1000, random_state=42)
    scores_combined = cross_val_score(clf_combined, X_combined_scaled, y, cv=cv, scoring='accuracy')

    results['combined'] = {
        'accuracy': scores_combined.mean(),
        'accuracy_std': scores_combined.std()
    }

    if print_output:
        print(f"\n--- Entropy + Best Layer Combined ---")
        print(f"Accuracy: {scores_combined.mean():.3f} (+/- {scores_combined.std():.3f})")

    # Summary
    improvement = scores_full.mean() - scores_entropy.mean()
    results['improvement'] = improvement

    if print_output:
        print(f"\n--- Summary ---")
        print(f"Entropy alone:        {scores_entropy.mean():.1%}")
        print(f"Best layer alone:     {best_layer_acc:.1%}")
        print(f"Full probe:           {scores_full.mean():.1%}")
        print(f"Entropy + best layer: {scores_combined.mean():.1%}")
        print(f"\nProbe improvement over entropy: +{improvement*100:.1f} percentage points")

    return results
