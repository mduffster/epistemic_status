"""
Cross-model comparison and generalization analysis.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

from .loader import ModelData, get_activation_matrix

POLICY_CATS = ['confident_incorrect', 'ambiguous', 'nonsensical']
FACTUAL_CATS = ['confident_correct', 'uncertain_correct', 'uncertain_incorrect']


def cross_model_generalization(
    train_data: ModelData,
    test_data: ModelData,
    position: str = 'last',
    layer: Optional[int] = None,
    print_output: bool = True
) -> Dict:
    """
    Train probe on one model, test on another.

    Tests whether epistemic representations are shared across models.

    Args:
        train_data: ModelData to train on
        test_data: ModelData to test on
        position: Token position
        layer: Specific layer or None for all

    Returns:
        Dictionary with generalization results
    """
    if print_output:
        print("\n" + "=" * 60)
        print("CROSS-MODEL GENERALIZATION")
        print(f"Train: {train_data.name} -> Test: {test_data.name}")
        print("=" * 60)

    results = {}

    # Get data
    X_train = get_activation_matrix(train_data, position, layer)
    y_train = train_data.df['correct'].values.astype(int)

    X_test = get_activation_matrix(test_data, position, layer)
    y_test = test_data.df['correct'].values.astype(int)

    # Check dimensions match
    if X_train.shape[1] != X_test.shape[1]:
        if print_output:
            print(f"Warning: Feature dimensions don't match ({X_train.shape[1]} vs {X_test.shape[1]})")
        return {'error': 'dimension_mismatch'}

    # Fit scaler on train, transform both
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train classifier
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train_scaled, y_train)

    # Evaluate on both
    train_acc = clf.score(X_train_scaled, y_train)
    test_acc = clf.score(X_test_scaled, y_test)

    train_probs = clf.predict_proba(X_train_scaled)[:, 1]
    test_probs = clf.predict_proba(X_test_scaled)[:, 1]

    train_auc = roc_auc_score(y_train, train_probs)
    test_auc = roc_auc_score(y_test, test_probs)

    results['train'] = {
        'accuracy': train_acc,
        'auc': train_auc,
        'n_samples': len(y_train)
    }

    results['test'] = {
        'accuracy': test_acc,
        'auc': test_auc,
        'n_samples': len(y_test)
    }

    results['generalization_gap'] = train_acc - test_acc
    results['auc_gap'] = train_auc - test_auc

    if print_output:
        print(f"\n--- Results ---")
        print(f"Train ({train_data.name}):")
        print(f"  Accuracy: {train_acc:.3f}")
        print(f"  AUC: {train_auc:.3f}")
        print(f"\nTest ({test_data.name}):")
        print(f"  Accuracy: {test_acc:.3f}")
        print(f"  AUC: {test_auc:.3f}")
        print(f"\nGeneralization gap: {results['generalization_gap']:.3f}")
        print(f"(Positive = worse on test set)")

    return results


def bidirectional_generalization(
    data1: ModelData,
    data2: ModelData,
    position: str = 'last',
    layer: Optional[int] = None,
    print_output: bool = True
) -> Dict:
    """
    Test generalization in both directions.

    Returns:
        Dictionary with results for both directions
    """
    if print_output:
        print("\n" + "=" * 60)
        print("BIDIRECTIONAL GENERALIZATION")
        print("=" * 60)

    results = {}

    # Direction 1: data1 -> data2
    results['forward'] = cross_model_generalization(
        data1, data2, position, layer, print_output=False
    )

    # Direction 2: data2 -> data1
    results['backward'] = cross_model_generalization(
        data2, data1, position, layer, print_output=False
    )

    if print_output:
        print(f"\n{data1.name} -> {data2.name}:")
        print(f"  Train acc: {results['forward']['train']['accuracy']:.3f}")
        print(f"  Test acc:  {results['forward']['test']['accuracy']:.3f}")

        print(f"\n{data2.name} -> {data1.name}:")
        print(f"  Train acc: {results['backward']['train']['accuracy']:.3f}")
        print(f"  Test acc:  {results['backward']['test']['accuracy']:.3f}")

        # Average transfer
        avg_transfer = (
            results['forward']['test']['accuracy'] +
            results['backward']['test']['accuracy']
        ) / 2
        print(f"\nAverage transfer accuracy: {avg_transfer:.3f}")

    return results


def compare_models(
    data1: ModelData,
    data2: ModelData,
    print_output: bool = True
) -> Dict:
    """
    Comprehensive comparison between two models.

    Returns:
        Dictionary with comparison results
    """
    if print_output:
        print("\n" + "=" * 60)
        print(f"MODEL COMPARISON: {data1.name} vs {data2.name}")
        print("=" * 60)

    results = {
        'models': [data1.name, data2.name],
        'basic': {},
        'entropy': {},
        'probing': {},
        'generalization': {}
    }

    # Basic stats comparison
    if print_output:
        print("\n--- Basic Statistics ---")

    for data in [data1, data2]:
        stats = {
            'accuracy': data.df['correct'].mean(),
            'mean_entropy': data.df['entropy'].mean(),
            'n_samples': len(data.df)
        }
        results['basic'][data.name] = stats

        if print_output:
            print(f"\n{data.name}:")
            print(f"  Accuracy: {stats['accuracy']:.3f}")
            print(f"  Mean entropy: {stats['mean_entropy']:.3f}")

    # Entropy comparison
    if print_output:
        print("\n--- Entropy by Correctness ---")

    for data in [data1, data2]:
        correct_ent = data.df[data.df['correct']]['entropy'].mean()
        incorrect_ent = data.df[~data.df['correct']]['entropy'].mean()
        results['entropy'][data.name] = {
            'correct': correct_ent,
            'incorrect': incorrect_ent,
            'delta': incorrect_ent - correct_ent
        }

        if print_output:
            print(f"\n{data.name}:")
            print(f"  Correct: {correct_ent:.3f}")
            print(f"  Incorrect: {incorrect_ent:.3f}")
            print(f"  Delta: {incorrect_ent - correct_ent:.3f}")

    # Probe accuracy (quick version - just last position, all layers)
    if print_output:
        print("\n--- Probe Accuracy (last position, all layers) ---")

    from sklearn.model_selection import cross_val_score, StratifiedKFold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for data in [data1, data2]:
        X = get_activation_matrix(data, 'last')
        y = data.df['correct'].values.astype(int)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        clf = LogisticRegression(max_iter=1000, random_state=42)
        scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='accuracy')

        results['probing'][data.name] = {
            'accuracy': scores.mean(),
            'std': scores.std()
        }

        if print_output:
            print(f"{data.name}: {scores.mean():.3f} (+/- {scores.std():.3f})")

    # Cross-model generalization
    results['generalization'] = bidirectional_generalization(
        data1, data2, print_output=print_output
    )

    return results


def layer_generalization(
    train_data: ModelData,
    test_data: ModelData,
    position: str = 'last',
    print_output: bool = True
) -> pd.DataFrame:
    """
    Test generalization at each layer.

    Returns:
        DataFrame with layer-wise generalization results
    """
    if print_output:
        print("\n--- Layer-wise Generalization ---")
        print(f"Train: {train_data.name} -> Test: {test_data.name}")

    n_layers = min(train_data.n_layers, test_data.n_layers)
    results = []

    for layer in range(n_layers):
        res = cross_model_generalization(
            train_data, test_data, position, layer, print_output=False
        )

        if 'error' not in res:
            results.append({
                'layer': layer,
                'train_acc': res['train']['accuracy'],
                'test_acc': res['test']['accuracy'],
                'gap': res['generalization_gap']
            })

            if print_output:
                print(f"Layer {layer:2d}: train={res['train']['accuracy']:.3f}, "
                      f"test={res['test']['accuracy']:.3f}, gap={res['generalization_gap']:.3f}")

    return pd.DataFrame(results)


def transfer_by_category(
    train_data: ModelData,
    test_data: ModelData,
    position: str = 'last',
    print_output: bool = True
) -> Dict:
    """
    Train probe on one model, test on another, broken down by category.

    Tests whether probe transfer differs for policy vs factual categories.
    This reveals whether fine-tuning selectively warps certain representations.

    Args:
        train_data: ModelData to train on (typically base model)
        test_data: ModelData to test on (typically instruct model)
        position: Token position
        print_output: Whether to print results

    Returns:
        Dictionary with transfer accuracy by category and aggregates
    """
    if print_output:
        print("\n" + "=" * 70)
        print("PROBE TRANSFER BY CATEGORY")
        print(f"Train: {train_data.name} -> Test: {test_data.name}")
        print("=" * 70)

    # Get activations
    X_train = get_activation_matrix(train_data, position)
    y_train = train_data.df['correct'].values.astype(int)

    X_test = get_activation_matrix(test_data, position)
    y_test = test_data.df['correct'].values.astype(int)
    cats_test = test_data.df['category'].values

    # Check dimensions
    if X_train.shape[1] != X_test.shape[1]:
        if print_output:
            print(f"Error: Feature dimensions don't match ({X_train.shape[1]} vs {X_test.shape[1]})")
        return {'error': 'dimension_mismatch'}

    # Train probe on train_data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train_scaled, y_train)

    # Overall accuracy
    train_acc = clf.score(X_train_scaled, y_train)
    test_acc = clf.score(X_test_scaled, y_test)

    # By category
    preds = clf.predict(X_test_scaled)

    results = {
        'train_model': train_data.name,
        'test_model': test_data.name,
        'overall_train': train_acc,
        'overall_test': test_acc,
        'by_category': {}
    }

    if print_output:
        print(f"\nOverall: Train={train_acc:.3f}, Test={test_acc:.3f}")
        print(f"\nBy category:")

    for cat in POLICY_CATS + FACTUAL_CATS:
        mask = cats_test == cat
        if mask.sum() > 0:
            cat_acc = (preds[mask] == y_test[mask]).mean()
            cat_n = mask.sum()
            results['by_category'][cat] = {'accuracy': cat_acc, 'n': int(cat_n)}

            if print_output:
                cat_type = "POLICY" if cat in POLICY_CATS else "FACTUAL"
                print(f"  {cat:<25} {cat_acc:.3f}  (n={cat_n}, {cat_type})")

    # Aggregate by type
    policy_accs = [results['by_category'][c]['accuracy'] for c in POLICY_CATS if c in results['by_category']]
    factual_accs = [results['by_category'][c]['accuracy'] for c in FACTUAL_CATS if c in results['by_category']]

    results['policy_mean'] = np.mean(policy_accs) if policy_accs else None
    results['factual_mean'] = np.mean(factual_accs) if factual_accs else None
    results['factual_policy_gap'] = results['factual_mean'] - results['policy_mean']

    if print_output:
        print(f"\n--- Summary ---")
        print(f"Policy categories mean:  {results['policy_mean']:.3f}")
        print(f"Factual categories mean: {results['factual_mean']:.3f}")
        print(f"Gap (factual - policy):  {results['factual_policy_gap']:+.3f}")

    return results
