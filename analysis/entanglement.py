"""
Entanglement analysis functions.

Tests the hypothesis that RLHF entangles representations for certain categories,
making them harder to distinguish via linear probing.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats

from .loader import ModelData, get_activation_matrix


CATEGORIES = ['confident_correct', 'confident_incorrect', 'uncertain_correct',
              'uncertain_incorrect', 'ambiguous', 'nonsensical']

# Categories expected to receive heavy RLHF treatment
RLHF_CATEGORIES = ['confident_incorrect', 'nonsensical', 'ambiguous']
NON_RLHF_CATEGORIES = ['confident_correct', 'uncertain_correct']


def probe_confidence_by_category(
    data: ModelData,
    position: str = 'last',
    test_size: float = 0.2,
    random_state: int = 42,
    print_output: bool = True
) -> Dict:
    """
    Measure probe confidence (max probability) by category.

    Lower confidence on a category suggests the probe is less certain,
    potentially indicating entangled representations.

    Args:
        data: ModelData object
        position: Token position ('first', 'middle', 'last')
        test_size: Fraction of data for test set
        random_state: Random seed
        print_output: Whether to print results

    Returns:
        Dictionary with confidence scores by category
    """
    X = get_activation_matrix(data, position)
    y = data.df['correct'].values.astype(int)
    categories = data.df['category'].values

    # Train/test split
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(y)), test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000, random_state=random_state)
    clf.fit(X_train_scaled, y_train)

    # Get probabilities and confidence
    probs = clf.predict_proba(X_test_scaled)
    confidence = np.max(probs, axis=1)

    # Get predictions and errors
    preds = clf.predict(X_test_scaled)
    errors = preds != y_test

    test_cats = categories[idx_test]

    results = {
        'by_category': {},
        'summary': {}
    }

    if print_output:
        print("\n" + "=" * 60)
        print("PROBE CONFIDENCE BY CATEGORY")
        print("=" * 60)
        print(f"{'Category':<25} {'Confidence':>12} {'Error Rate':>12} {'N':>6}")
        print("-" * 55)

    for cat in CATEGORIES:
        mask = test_cats == cat
        if mask.sum() > 0:
            cat_conf = confidence[mask].mean()
            cat_err = errors[mask].mean()
            cat_n = mask.sum()

            results['by_category'][cat] = {
                'confidence': cat_conf,
                'error_rate': cat_err,
                'n_samples': int(cat_n)
            }

            if print_output:
                print(f"{cat:<25} {cat_conf:>12.3f} {cat_err:>12.3f} {cat_n:>6}")

    # Summary: RLHF vs non-RLHF categories
    rlhf_conf = np.mean([results['by_category'][c]['confidence']
                         for c in RLHF_CATEGORIES if c in results['by_category']])
    non_rlhf_conf = np.mean([results['by_category'][c]['confidence']
                             for c in NON_RLHF_CATEGORIES if c in results['by_category']])

    results['summary']['rlhf_mean_confidence'] = rlhf_conf
    results['summary']['non_rlhf_mean_confidence'] = non_rlhf_conf
    results['summary']['confidence_gap'] = non_rlhf_conf - rlhf_conf

    if print_output:
        print("-" * 55)
        print(f"RLHF categories mean:     {rlhf_conf:.3f}")
        print(f"Non-RLHF categories mean: {non_rlhf_conf:.3f}")
        print(f"Gap (non-RLHF - RLHF):    {non_rlhf_conf - rlhf_conf:+.3f}")

    return results


def probe_confidence_layerwise(
    data: ModelData,
    position: str = 'last',
    test_size: float = 0.2,
    random_state: int = 42,
    print_output: bool = True
) -> pd.DataFrame:
    """
    Measure probe confidence by category at each layer.

    Shows where in the network entanglement emerges.

    Args:
        data: ModelData object
        position: Token position
        test_size: Fraction for test set
        random_state: Random seed
        print_output: Whether to print results

    Returns:
        DataFrame with confidence by layer and category
    """
    y = data.df['correct'].values.astype(int)
    categories = data.df['category'].values
    n_layers = data.n_layers

    # Get full activation tensor
    X_full = data.activations[f'resid_{position}']

    # Train/test split (same indices for all layers)
    indices = np.arange(len(y))
    idx_train, idx_test, y_train, y_test = train_test_split(
        indices, y, test_size=test_size, random_state=random_state, stratify=y
    )

    test_cats = categories[idx_test]

    results = []

    if print_output:
        print("\n" + "=" * 60)
        print("LAYER-WISE PROBE CONFIDENCE")
        print("=" * 60)

    for layer in range(n_layers):
        X_layer = X_full[:, layer, :]
        X_train = X_layer[idx_train]
        X_test = X_layer[idx_test]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=1000, random_state=random_state)
        clf.fit(X_train_scaled, y_train)

        probs = clf.predict_proba(X_test_scaled)
        confidence = np.max(probs, axis=1)

        row = {'layer': layer}
        for cat in CATEGORIES:
            mask = test_cats == cat
            if mask.sum() > 0:
                row[cat] = confidence[mask].mean()
        results.append(row)

    df = pd.DataFrame(results).set_index('layer')

    if print_output:
        # Show subset of layers
        display_layers = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]
        print(df.loc[display_layers].round(3).to_string())
        print(f"\n(Showing layers {display_layers})")

    return df


def bootstrap_confidence_intervals(
    data: ModelData,
    position: str = 'last',
    n_bootstrap: int = 100,
    test_size: float = 0.2,
    confidence_level: float = 0.95,
    print_output: bool = True
) -> Dict:
    """
    Compute bootstrap confidence intervals for probe error rates by category.

    Provides statistical rigor for error rate comparisons.

    Args:
        data: ModelData object
        position: Token position
        n_bootstrap: Number of bootstrap iterations
        test_size: Fraction for test set
        confidence_level: CI level (e.g., 0.95 for 95% CI)
        print_output: Whether to print results

    Returns:
        Dictionary with CIs for each category
    """
    X = get_activation_matrix(data, position)
    y = data.df['correct'].values.astype(int)
    categories = data.df['category'].values

    if print_output:
        print("\n" + "=" * 60)
        print(f"BOOTSTRAP CONFIDENCE INTERVALS (n={n_bootstrap})")
        print("=" * 60)

    # Store error rates for each bootstrap
    bootstrap_errors = {cat: [] for cat in CATEGORIES}

    for i in range(n_bootstrap):
        # Different random state for each bootstrap
        rs = 42 + i

        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, np.arange(len(y)), test_size=test_size, random_state=rs, stratify=y
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=1000, random_state=rs)
        clf.fit(X_train_scaled, y_train)

        preds = clf.predict(X_test_scaled)
        errors = preds != y_test
        test_cats = categories[idx_test]

        for cat in CATEGORIES:
            mask = test_cats == cat
            if mask.sum() > 0:
                bootstrap_errors[cat].append(errors[mask].mean())

    # Compute CIs
    alpha = 1 - confidence_level
    results = {}

    if print_output:
        print(f"\n{'Category':<25} {'Mean':>8} {'Std':>8} {'CI Low':>8} {'CI High':>8}")
        print("-" * 60)

    for cat in CATEGORIES:
        if bootstrap_errors[cat]:
            errors = np.array(bootstrap_errors[cat])
            mean = errors.mean()
            std = errors.std()
            ci_low = np.percentile(errors, alpha/2 * 100)
            ci_high = np.percentile(errors, (1 - alpha/2) * 100)

            results[cat] = {
                'mean': mean,
                'std': std,
                'ci_low': ci_low,
                'ci_high': ci_high,
                'n_bootstrap': len(errors)
            }

            if print_output:
                print(f"{cat:<25} {mean:>8.3f} {std:>8.3f} {ci_low:>8.3f} {ci_high:>8.3f}")

    return results


def held_out_category_generalization(
    data: ModelData,
    position: str = 'last',
    n_folds: int = 5,
    random_state: int = 42,
    print_output: bool = True
) -> Dict:
    """
    Test generalization by training probe excluding each category, then testing on it.

    If entanglement is real, RLHF categories should generalize worse when held out.

    Args:
        data: ModelData object
        position: Token position
        n_folds: Number of CV folds for training
        random_state: Random seed
        print_output: Whether to print results

    Returns:
        Dictionary with held-out performance for each category
    """
    X = get_activation_matrix(data, position)
    y = data.df['correct'].values.astype(int)
    categories = data.df['category'].values

    if print_output:
        print("\n" + "=" * 60)
        print("HELD-OUT CATEGORY GENERALIZATION")
        print("=" * 60)
        print("Train on all other categories, test on held-out category")
        print(f"\n{'Held-out Category':<25} {'Accuracy':>10} {'AUC':>10} {'N':>6}")
        print("-" * 55)

    results = {}

    for held_out_cat in CATEGORIES:
        # Split data
        held_out_mask = categories == held_out_cat
        train_mask = ~held_out_mask

        if held_out_mask.sum() < 5:
            continue

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[held_out_mask], y[held_out_mask]

        # Check if test set has both classes
        if len(np.unique(y_test)) < 2:
            # Can still compute accuracy but not AUC
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            clf = LogisticRegression(max_iter=1000, random_state=random_state)
            clf.fit(X_train_scaled, y_train)

            preds = clf.predict(X_test_scaled)
            acc = (preds == y_test).mean()
            auc = np.nan
        else:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            clf = LogisticRegression(max_iter=1000, random_state=random_state)
            clf.fit(X_train_scaled, y_train)

            preds = clf.predict(X_test_scaled)
            probs = clf.predict_proba(X_test_scaled)[:, 1]

            acc = (preds == y_test).mean()
            auc = roc_auc_score(y_test, probs)

        results[held_out_cat] = {
            'accuracy': acc,
            'auc': auc,
            'n_samples': int(held_out_mask.sum()),
            'is_rlhf_category': held_out_cat in RLHF_CATEGORIES
        }

        if print_output:
            auc_str = f"{auc:.3f}" if not np.isnan(auc) else "N/A"
            print(f"{held_out_cat:<25} {acc:>10.3f} {auc_str:>10} {held_out_mask.sum():>6}")

    # Summary
    rlhf_acc = np.mean([results[c]['accuracy'] for c in RLHF_CATEGORIES if c in results])
    non_rlhf_acc = np.mean([results[c]['accuracy'] for c in NON_RLHF_CATEGORIES if c in results])

    results['summary'] = {
        'rlhf_mean_accuracy': rlhf_acc,
        'non_rlhf_mean_accuracy': non_rlhf_acc,
        'accuracy_gap': non_rlhf_acc - rlhf_acc
    }

    if print_output:
        print("-" * 55)
        print(f"RLHF categories mean acc:     {rlhf_acc:.3f}")
        print(f"Non-RLHF categories mean acc: {non_rlhf_acc:.3f}")
        print(f"Gap (non-RLHF - RLHF):        {non_rlhf_acc - rlhf_acc:+.3f}")

    return results


def compare_base_instruct_entanglement(
    base_data: ModelData,
    instruct_data: ModelData,
    position: str = 'last',
    n_bootstrap: int = 50,
    print_output: bool = True
) -> Dict:
    """
    Compare entanglement metrics between base and instruct models.

    Args:
        base_data: ModelData for base model
        instruct_data: ModelData for instruct model
        position: Token position
        n_bootstrap: Bootstrap iterations for CIs
        print_output: Whether to print results

    Returns:
        Dictionary with comparison results
    """
    if print_output:
        print("\n" + "=" * 70)
        print("BASE vs INSTRUCT ENTANGLEMENT COMPARISON")
        print("=" * 70)

    # Get probe confidence for both
    base_conf = probe_confidence_by_category(base_data, position, print_output=False)
    inst_conf = probe_confidence_by_category(instruct_data, position, print_output=False)

    # Get bootstrap CIs for both
    base_ci = bootstrap_confidence_intervals(base_data, position, n_bootstrap, print_output=False)
    inst_ci = bootstrap_confidence_intervals(instruct_data, position, n_bootstrap, print_output=False)

    results = {
        'base': base_conf,
        'instruct': inst_conf,
        'base_ci': base_ci,
        'instruct_ci': inst_ci,
        'deltas': {}
    }

    if print_output:
        print(f"\n{'Category':<20} {'Base Err':>10} {'Inst Err':>10} {'Delta':>10} {'Significant':>12}")
        print("-" * 65)

    for cat in CATEGORIES:
        if cat in base_conf['by_category'] and cat in inst_conf['by_category']:
            base_err = base_conf['by_category'][cat]['error_rate']
            inst_err = inst_conf['by_category'][cat]['error_rate']
            delta = inst_err - base_err

            # Check significance using bootstrap CIs
            # If CIs don't overlap, consider significant
            base_ci_high = base_ci[cat]['ci_high'] if cat in base_ci else base_err
            inst_ci_low = inst_ci[cat]['ci_low'] if cat in inst_ci else inst_err
            significant = inst_ci_low > base_ci_high  # Instruct significantly higher

            results['deltas'][cat] = {
                'base_error': base_err,
                'instruct_error': inst_err,
                'delta': delta,
                'significant': significant
            }

            if print_output:
                sig_str = "YES" if significant else "no"
                print(f"{cat:<20} {base_err:>10.3f} {inst_err:>10.3f} {delta:>+10.3f} {sig_str:>12}")

    if print_output:
        # Summary
        rlhf_deltas = [results['deltas'][c]['delta'] for c in RLHF_CATEGORIES if c in results['deltas']]
        non_rlhf_deltas = [results['deltas'][c]['delta'] for c in NON_RLHF_CATEGORIES if c in results['deltas']]

        print("-" * 65)
        print(f"RLHF categories mean delta:     {np.mean(rlhf_deltas):+.3f}")
        print(f"Non-RLHF categories mean delta: {np.mean(non_rlhf_deltas):+.3f}")

    return results


def run_full_entanglement_analysis(
    data: ModelData,
    position: str = 'last',
    n_bootstrap: int = 50,
    print_output: bool = True
) -> Dict:
    """
    Run all entanglement analyses on a single model.

    Args:
        data: ModelData object
        position: Token position
        n_bootstrap: Bootstrap iterations
        print_output: Whether to print results

    Returns:
        Dictionary with all analysis results
    """
    if print_output:
        print("\n" + "=" * 70)
        print(f"FULL ENTANGLEMENT ANALYSIS: {data.name}")
        print("=" * 70)

    results = {
        'model': data.name,
        'probe_confidence': probe_confidence_by_category(data, position, print_output=print_output),
        'layerwise': probe_confidence_layerwise(data, position, print_output=print_output),
        'bootstrap_ci': bootstrap_confidence_intervals(data, position, n_bootstrap, print_output=print_output),
        'held_out': held_out_category_generalization(data, position, print_output=print_output),
        'activation_similarity': activation_similarity_by_category(data, position, print_output=print_output)
    }

    return results


def activation_similarity_by_category(
    data: ModelData,
    position: str = 'last',
    print_output: bool = True
) -> Dict:
    """
    Compute pairwise cosine similarity between category centroids.

    Shows how similar different categories are in activation space.

    Args:
        data: ModelData object
        position: Token position
        print_output: Whether to print results

    Returns:
        Dictionary with pairwise similarities
    """
    X = get_activation_matrix(data, position)

    # Compute category centroids
    centroids = {}
    for cat in CATEGORIES:
        mask = data.df['category'] == cat
        if mask.sum() > 0:
            centroids[cat] = X[mask].mean(axis=0).reshape(1, -1)

    # Compute pairwise similarities
    results = {'centroids': {}, 'pairwise': {}}

    for cat in CATEGORIES:
        if cat in centroids:
            results['centroids'][cat] = centroids[cat].flatten()

    if print_output:
        print("\n" + "=" * 60)
        print("ACTIVATION SIMILARITY BY CATEGORY")
        print("=" * 60)
        print("\nPairwise cosine similarity between category centroids:")
        print()

    # Create similarity matrix
    cats_present = [c for c in CATEGORIES if c in centroids]
    n_cats = len(cats_present)

    if print_output:
        # Header
        print(f"{'':>12}", end="")
        for cat in cats_present:
            print(f"{cat[:8]:>10}", end="")
        print()
        print("-" * (12 + 10 * n_cats))

    for cat1 in cats_present:
        if print_output:
            print(f"{cat1[:12]:<12}", end="")
        for cat2 in cats_present:
            sim = cosine_similarity(centroids[cat1], centroids[cat2])[0, 0]
            results['pairwise'][f"{cat1}_{cat2}"] = sim
            if print_output:
                print(f"{sim:>10.3f}", end="")
        if print_output:
            print()

    return results


def compare_activation_similarity(
    base_data: ModelData,
    instruct_data: ModelData,
    position: str = 'last',
    print_output: bool = True
) -> Dict:
    """
    Compare activation similarity between base and instruct models.

    Tests whether confident_incorrect moves toward uncertain_correct after RLHF.

    Args:
        base_data: ModelData for base model
        instruct_data: ModelData for instruct model
        position: Token position
        print_output: Whether to print results

    Returns:
        Dictionary with similarity comparison
    """
    base_sim = activation_similarity_by_category(base_data, position, print_output=False)
    inst_sim = activation_similarity_by_category(instruct_data, position, print_output=False)

    results = {
        'base': base_sim,
        'instruct': inst_sim,
        'deltas': {},
        'direction_analysis': {}
    }

    if print_output:
        print("\n" + "=" * 70)
        print("ACTIVATION SIMILARITY COMPARISON (base vs instruct)")
        print("=" * 70)

    # Key comparisons
    key_pairs = [
        ('confident_incorrect', 'confident_correct', 'ci_vs_cc'),
        ('confident_incorrect', 'uncertain_correct', 'ci_vs_uc'),
        ('confident_incorrect', 'nonsensical', 'ci_vs_ns'),
        ('ambiguous', 'uncertain_correct', 'amb_vs_uc'),
    ]

    if print_output:
        print(f"\n{'Comparison':<35} {'Base':>10} {'Instruct':>10} {'Delta':>10}")
        print("-" * 65)

    for cat1, cat2, label in key_pairs:
        key = f"{cat1}_{cat2}"
        base_val = base_sim['pairwise'].get(key, np.nan)
        inst_val = inst_sim['pairwise'].get(key, np.nan)
        delta = inst_val - base_val

        results['deltas'][label] = {
            'base': base_val,
            'instruct': inst_val,
            'delta': delta
        }

        if print_output:
            print(f"{cat1[:15]} vs {cat2[:15]:<15} {base_val:>10.3f} {inst_val:>10.3f} {delta:>+10.3f}")

    # Direction analysis for confident_incorrect
    ci_cc_key = 'confident_incorrect_confident_correct'
    ci_uc_key = 'confident_incorrect_uncertain_correct'

    toward_uncertain = inst_sim['pairwise'].get(ci_uc_key, 0) - base_sim['pairwise'].get(ci_uc_key, 0)
    away_confident = base_sim['pairwise'].get(ci_cc_key, 0) - inst_sim['pairwise'].get(ci_cc_key, 0)
    net_direction = toward_uncertain + away_confident

    results['direction_analysis'] = {
        'toward_uncertain': toward_uncertain,
        'away_from_confident': away_confident,
        'net_direction': net_direction,
        'interpretation': 'toward_uncertainty' if net_direction > 0 else 'toward_confidence'
    }

    if print_output:
        print("\n" + "-" * 65)
        print("Direction analysis for confident_incorrect:")
        print(f"  → toward uncertain_correct:    {toward_uncertain:+.3f}")
        print(f"  ← away from confident_correct: {away_confident:+.3f}")
        print(f"  Net direction:                 {net_direction:+.3f} ({results['direction_analysis']['interpretation']})")

    return results
