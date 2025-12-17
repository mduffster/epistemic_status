"""
Linear probing functions for epistemic analysis.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .loader import ModelData, get_activation_matrix


def prepare_probe_data(
    data: ModelData,
    target: str = 'correct',
    position: str = 'last',
    layer: Optional[int] = None,
    categories: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare data for linear probing.

    Args:
        data: ModelData object
        target: What to predict ('correct', 'category', 'acknowledged_unknown')
        position: Token position ('first', 'middle', 'last')
        layer: Specific layer or None for all layers
        categories: Subset of categories or None for all

    Returns:
        X: Feature matrix
        y: Labels
        mask: Boolean mask for filtering
    """
    X = get_activation_matrix(data, position, layer)

    # Filter by categories if specified
    if categories is not None:
        mask = data.df['category'].isin(categories).values
        X = X[mask]
        df_filtered = data.df[mask]
    else:
        mask = np.ones(len(data.df), dtype=bool)
        df_filtered = data.df

    # Get labels
    if target == 'correct':
        y = df_filtered['correct'].values.astype(int)
    elif target == 'category':
        y = pd.Categorical(df_filtered['category']).codes
    elif target == 'acknowledged_unknown':
        y = df_filtered['acknowledged_unknown'].values.astype(int)
    else:
        raise ValueError(f"Unknown target: {target}")

    return X, y, mask


def run_linear_probe(
    data: ModelData,
    target: str = 'correct',
    position: str = 'last',
    layer: Optional[int] = None,
    categories: Optional[List[str]] = None,
    n_folds: int = 5,
    print_output: bool = True
) -> Optional[Dict]:
    """
    Run linear probe with cross-validation.

    Returns:
        Dictionary with results or None if insufficient data
    """
    X, y, _ = prepare_probe_data(data, target, position, layer, categories)

    # Check for class balance
    unique, counts = np.unique(y, return_counts=True)
    if print_output:
        print(f"\nClass distribution: {dict(zip(unique, counts))}")

    # Skip if only one class
    if len(unique) < 2:
        if print_output:
            print("Skipping: Only one class present in data")
        return None

    # Adjust folds if needed
    if min(counts) < n_folds:
        if print_output:
            print(f"Warning: Too few samples in minority class for {n_folds}-fold CV")
        n_folds = min(counts)

    # Standardize and run CV
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='accuracy')

    results = {
        'target': target,
        'position': position,
        'layer': layer,
        'categories': categories,
        'n_samples': len(y),
        'n_features': X.shape[1],
        'accuracy_mean': scores.mean(),
        'accuracy_std': scores.std(),
        'fold_scores': scores
    }

    if print_output:
        print(f"Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")

    return results


def layer_analysis(
    data: ModelData,
    target: str = 'correct',
    position: str = 'last',
    categories: Optional[List[str]] = None,
    print_output: bool = True
) -> pd.DataFrame:
    """
    Run probes at each layer to see where information emerges.

    Returns:
        DataFrame with layer-wise results
    """
    n_layers = data.n_layers
    results = []

    if print_output:
        print(f"\n--- Layer-wise Probe for {target} ---")

    for layer in range(n_layers):
        res = run_linear_probe(
            data, target, position, layer, categories,
            n_folds=5, print_output=False
        )
        if res:
            res['layer'] = layer
            results.append(res)
            if print_output:
                print(f"Layer {layer:2d}: {res['accuracy_mean']:.3f}")

    return pd.DataFrame(results)


def compare_positions(
    data: ModelData,
    target: str = 'correct',
    layer: Optional[int] = None,
    categories: Optional[List[str]] = None,
    print_output: bool = True
) -> Dict[str, Dict]:
    """
    Compare probe accuracy across token positions.

    Returns:
        Dictionary mapping position to results
    """
    results = {}

    if print_output:
        print(f"\n--- Position Comparison for {target} ---")

    for position in ['first', 'middle', 'last']:
        res = run_linear_probe(
            data, target, position, layer, categories,
            print_output=print_output
        )
        results[position] = res
        if print_output and res:
            print(f"{position}: {res['accuracy_mean']:.3f} (+/- {res['accuracy_std']:.3f})")

    return results


def probe_with_controls(
    data: ModelData,
    target: str = 'correct',
    position: str = 'last',
    exclude_categories: Optional[List[str]] = None,
    exclude_features: Optional[List[str]] = None,
    print_output: bool = True
) -> Optional[Dict]:
    """
    Run linear probe with controls for prompt features.

    Args:
        data: ModelData object
        target: What to predict
        position: Token position
        exclude_categories: Categories to exclude
        exclude_features: Exclude prompts with these features

    Returns:
        Dictionary with results or None
    """
    # Create mask
    mask = pd.Series([True] * len(data.df))

    if exclude_categories:
        mask &= ~data.df['category'].isin(exclude_categories)

    if exclude_features:
        for feat in exclude_features:
            if feat in data.df.columns:
                mask &= ~data.df[feat]

    indices = data.df[mask].index.tolist()

    if print_output:
        print(f"\nControlled probe: {len(indices)} samples after filtering")

    # Get activations for these indices
    X = get_activation_matrix(data, position)
    X = X[mask.values]
    y = data.df.loc[mask, 'correct'].values.astype(int)

    # Check class balance
    unique, counts = np.unique(y, return_counts=True)
    if print_output:
        print(f"Class distribution: {dict(zip(unique, counts))}")

    if len(unique) < 2:
        if print_output:
            print("Skipping: Only one class present")
        return None

    if min(counts) < 5:
        if print_output:
            print("Warning: Too few samples for reliable CV")
        return None

    # Standardize and run CV
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    n_folds = min(5, min(counts))
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='accuracy')

    if print_output:
        print(f"Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")

    return {
        'n_samples': len(indices),
        'accuracy_mean': scores.mean(),
        'accuracy_std': scores.std()
    }
