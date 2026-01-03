"""
Entanglement analysis functions.

Tests the hypothesis that RLHF entangles representations for certain categories,
making them harder to distinguish via linear probing.
"""

from typing import Dict, List, Optional, Tuple
import gc
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats

from .loader import ModelData, get_activation_matrix


def _cleanup():
    """Force garbage collection."""
    gc.collect()


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

        # Cleanup after each iteration to prevent memory accumulation
        del X_train, X_test, y_train, y_test, idx_train, idx_test
        del X_train_scaled, X_test_scaled, scaler, clf, preds, errors, test_cats
        if i % 10 == 0:
            _cleanup()

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

    # Process base model first, then cleanup
    base_conf = probe_confidence_by_category(base_data, position, print_output=False)
    _cleanup()
    base_ci = bootstrap_confidence_intervals(base_data, position, n_bootstrap, print_output=False)
    _cleanup()

    # Process instruct model
    inst_conf = probe_confidence_by_category(instruct_data, position, print_output=False)
    _cleanup()
    inst_ci = bootstrap_confidence_intervals(instruct_data, position, n_bootstrap, print_output=False)
    _cleanup()

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


def compare_activation_similarity_legacy(
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


# Alias for backwards compatibility
compare_activation_similarity = compare_activation_similarity_legacy


# =============================================================================
# RIGOROUS STATISTICAL TESTING (NEW)
# =============================================================================

from .statistics import (
    permutation_test,
    bootstrap_paired_test,
    probe_permutation_test,
    correct_multiple_comparisons,
    run_seed_sensitivity,
    summarize_significance,
    SignificanceResult,
)


def test_entanglement_significance(
    base_data: ModelData,
    instruct_data: ModelData,
    position: str = 'last',
    n_permutations: int = 1000,
    n_bootstrap: int = 1000,
    correction_method: str = 'fdr_bh',
    print_output: bool = True
) -> Dict:
    """
    Rigorous significance testing for entanglement claims.

    Tests:
    1. Is there a significant difference in probe error between base/instruct per category?
    2. Is RLHF category degradation significantly greater than non-RLHF?
    3. Multiple comparison correction across all tests

    Args:
        base_data: ModelData for base model
        instruct_data: ModelData for instruct model
        position: Token position
        n_permutations: Number of permutations for significance tests
        n_bootstrap: Number of bootstrap iterations
        correction_method: 'bonferroni', 'holm', 'fdr_bh', or 'fdr_by'
        print_output: Whether to print results

    Returns:
        Dictionary with significance test results
    """
    if print_output:
        print("\n" + "=" * 70)
        print("ENTANGLEMENT SIGNIFICANCE TESTING")
        print("=" * 70)
        print(f"Base: {base_data.name} | Instruct: {instruct_data.name}")
        print(f"Permutations: {n_permutations} | Bootstrap: {n_bootstrap}")
        print(f"Correction: {correction_method}")

    results = {
        'per_category': {},
        'group_comparison': {},
        'corrected_results': None
    }

    # Get probe error rates for each category across multiple seeds
    base_errors_by_cat = _get_category_errors_multi_seed(
        base_data, position, n_seeds=n_bootstrap
    )
    inst_errors_by_cat = _get_category_errors_multi_seed(
        instruct_data, position, n_seeds=n_bootstrap
    )

    # Test 1: Per-category significance (base vs instruct error rate)
    if print_output:
        print(f"\n--- Per-Category Tests (Base vs Instruct Error Rate) ---")
        print(f"{'Category':<20} {'Base Err':>10} {'Inst Err':>10} {'Diff':>10} {'p-value':>10} {'Sig':>5}")
        print("-" * 70)

    all_tests = {}

    for cat in CATEGORIES:
        if cat in base_errors_by_cat and cat in inst_errors_by_cat:
            base_errs = np.array(base_errors_by_cat[cat])
            inst_errs = np.array(inst_errors_by_cat[cat])

            # Paired bootstrap test (same seeds, different models)
            sig_result = bootstrap_paired_test(
                inst_errs, base_errs,  # inst - base (positive = instruct worse)
                n_bootstrap=n_bootstrap,
                random_state=42
            )

            results['per_category'][cat] = {
                'base_mean': base_errs.mean(),
                'instruct_mean': inst_errs.mean(),
                'difference': sig_result.statistic,
                'p_value': sig_result.p_value,
                'ci_low': sig_result.ci_low,
                'ci_high': sig_result.ci_high,
                'effect_size': sig_result.effect_size,
                'significant_uncorrected': sig_result.significant
            }

            all_tests[f'cat_{cat}'] = sig_result

            if print_output:
                sig_str = "*" if sig_result.p_value < 0.05 else ""
                print(f"{cat:<20} {base_errs.mean():>10.3f} {inst_errs.mean():>10.3f} "
                      f"{sig_result.statistic:>+10.3f} {sig_result.p_value:>10.4f} {sig_str:>5}")

    # Test 2: RLHF vs Non-RLHF group comparison
    if print_output:
        print(f"\n--- Group Comparison (RLHF vs Non-RLHF Categories) ---")

    # Compute deltas (instruct - base error) for each category
    rlhf_deltas = []
    non_rlhf_deltas = []

    for cat in CATEGORIES:
        if cat in results['per_category']:
            delta = results['per_category'][cat]['difference']
            if cat in RLHF_CATEGORIES:
                rlhf_deltas.append(delta)
            elif cat in NON_RLHF_CATEGORIES:
                non_rlhf_deltas.append(delta)

    if len(rlhf_deltas) > 0 and len(non_rlhf_deltas) > 0:
        rlhf_deltas = np.array(rlhf_deltas)
        non_rlhf_deltas = np.array(non_rlhf_deltas)

        # Permutation test: is RLHF degradation > non-RLHF degradation?
        group_test = permutation_test(
            rlhf_deltas, non_rlhf_deltas,
            statistic='mean_diff',
            n_permutations=n_permutations,
            alternative='greater',  # RLHF expected to be higher
            random_state=42
        )

        results['group_comparison'] = {
            'rlhf_mean_delta': rlhf_deltas.mean(),
            'non_rlhf_mean_delta': non_rlhf_deltas.mean(),
            'group_difference': group_test.statistic,
            'p_value': group_test.p_value,
            'effect_size': group_test.effect_size,
            'significant': group_test.significant
        }

        all_tests['rlhf_vs_nonrlhf'] = group_test

        if print_output:
            print(f"RLHF categories mean Δ:     {rlhf_deltas.mean():+.4f}")
            print(f"Non-RLHF categories mean Δ: {non_rlhf_deltas.mean():+.4f}")
            print(f"Difference (RLHF - NonRLHF): {group_test.statistic:+.4f}")
            print(f"p-value (one-sided):         {group_test.p_value:.4f}")
            print(f"Effect size (Cohen's d):     {group_test.effect_size:.3f}")
            sig_str = "SIGNIFICANT" if group_test.significant else "not significant"
            print(f"Result: {sig_str} at α=0.05")

    # Test 3: Multiple comparison correction
    if print_output:
        print(f"\n--- Multiple Comparison Correction ({correction_method}) ---")

    p_values = np.array([t.p_value for t in all_tests.values()])
    labels = list(all_tests.keys())

    corrected = correct_multiple_comparisons(
        p_values, method=correction_method, labels=labels, print_output=print_output
    )

    results['corrected_results'] = {
        'method': correction_method,
        'n_tests': corrected.n_tests,
        'n_significant': corrected.n_significant,
        'corrected_p_values': dict(zip(labels, corrected.corrected_p_values)),
        'significant_after_correction': dict(zip(labels, corrected.significant))
    }

    # Summary
    if print_output:
        print(f"\n--- Summary ---")
        n_sig_uncorrected = sum(1 for t in all_tests.values() if t.significant)
        print(f"Tests significant before correction: {n_sig_uncorrected}/{len(all_tests)}")
        print(f"Tests significant after correction:  {corrected.n_significant}/{corrected.n_tests}")

    return results


def test_category_probe_difference(
    data: ModelData,
    category1: str,
    category2: str,
    position: str = 'last',
    n_permutations: int = 1000,
    print_output: bool = True
) -> SignificanceResult:
    """
    Test if probe performance differs significantly between two categories.

    Args:
        data: ModelData object
        category1: First category name
        category2: Second category name
        position: Token position
        n_permutations: Number of permutations
        print_output: Whether to print results

    Returns:
        SignificanceResult for the comparison
    """
    X = get_activation_matrix(data, position)
    y = data.df['correct'].values.astype(int)
    categories = data.df['category'].values

    result = probe_permutation_test(
        X, y, categories, category1, category2,
        metric='accuracy',
        n_permutations=n_permutations,
        random_state=42
    )

    if print_output:
        print(f"\n--- Probe Difference: {category1} vs {category2} ---")
        print(f"Observed difference: {result.statistic:+.4f}")
        print(f"p-value: {result.p_value:.4f}")
        print(f"95% CI: [{result.ci_low:.4f}, {result.ci_high:.4f}]")
        sig_str = "SIGNIFICANT" if result.significant else "not significant"
        print(f"Result: {sig_str}")

    return result


def run_entanglement_seed_sensitivity(
    base_data: ModelData,
    instruct_data: ModelData,
    position: str = 'last',
    n_seeds: int = 5,
    print_output: bool = True
) -> Dict:
    """
    Run seed sensitivity analysis for entanglement metrics.

    Tests whether key findings are stable across random seeds.

    Args:
        base_data: ModelData for base model
        instruct_data: ModelData for instruct model
        position: Token position
        n_seeds: Number of seeds to test
        print_output: Whether to print results

    Returns:
        Dictionary with sensitivity results
    """
    if print_output:
        print("\n" + "=" * 70)
        print("SEED SENSITIVITY ANALYSIS")
        print("=" * 70)
        print(f"Testing stability with {n_seeds} random seeds")

    X_base = get_activation_matrix(base_data, position)
    y_base = base_data.df['correct'].values.astype(int)

    X_inst = get_activation_matrix(instruct_data, position)
    y_inst = instruct_data.df['correct'].values.astype(int)

    # Test probe accuracy stability
    if print_output:
        print("\n--- Base Model Probe Stability ---")

    from .statistics import run_probe_seed_sensitivity
    base_sensitivity = run_probe_seed_sensitivity(
        X_base, y_base, n_seeds=n_seeds, print_output=print_output
    )

    if print_output:
        print("\n--- Instruct Model Probe Stability ---")

    inst_sensitivity = run_probe_seed_sensitivity(
        X_inst, y_inst, n_seeds=n_seeds, print_output=print_output
    )

    # Test entanglement metric stability
    if print_output:
        print("\n--- Entanglement Metrics Stability ---")

    def compute_entanglement_metrics(random_state, print_output=False):
        """Compute key entanglement metrics for a given seed."""
        # Get error rates per category for base
        base_errs = {}
        inst_errs = {}

        for cat in CATEGORIES:
            base_mask = base_data.df['category'] == cat
            inst_mask = instruct_data.df['category'] == cat

            if base_mask.sum() > 10 and inst_mask.sum() > 10:
                try:
                    # Train probe on base, get error on category
                    X_b = X_base[base_mask]
                    y_b = y_base[base_mask]

                    # Check we have enough samples of each class for stratified split
                    unique, counts = np.unique(y_b, return_counts=True)
                    if len(unique) < 2 or np.min(counts) < 3:
                        continue

                    from sklearn.model_selection import train_test_split
                    X_tr, X_te, y_tr, y_te = train_test_split(
                        X_b, y_b, test_size=0.3, random_state=random_state, stratify=y_b
                    )

                    scaler = StandardScaler()
                    X_tr_s = scaler.fit_transform(X_tr)
                    X_te_s = scaler.transform(X_te)

                    clf = LogisticRegression(max_iter=1000, random_state=random_state)
                    clf.fit(X_tr_s, y_tr)
                    base_errs[cat] = 1 - clf.score(X_te_s, y_te)

                    # Same for instruct
                    X_i = X_inst[inst_mask]
                    y_i = y_inst[inst_mask]

                    # Check we have enough samples of each class
                    unique, counts = np.unique(y_i, return_counts=True)
                    if len(unique) < 2 or np.min(counts) < 3:
                        continue

                    X_tr, X_te, y_tr, y_te = train_test_split(
                        X_i, y_i, test_size=0.3, random_state=random_state, stratify=y_i
                    )

                    scaler = StandardScaler()
                    X_tr_s = scaler.fit_transform(X_tr)
                    X_te_s = scaler.transform(X_te)

                    clf = LogisticRegression(max_iter=1000, random_state=random_state)
                    clf.fit(X_tr_s, y_tr)
                    inst_errs[cat] = 1 - clf.score(X_te_s, y_te)
                except Exception:
                    # Skip this category if there's any error
                    continue

        # Compute summary metrics
        rlhf_available = [c for c in RLHF_CATEGORIES if c in inst_errs and c in base_errs]
        nonrlhf_available = [c for c in NON_RLHF_CATEGORIES if c in inst_errs and c in base_errs]

        if not rlhf_available or not nonrlhf_available:
            return {'rlhf_delta': np.nan, 'nonrlhf_delta': np.nan, 'delta_gap': np.nan}

        rlhf_delta = np.mean([inst_errs[c] - base_errs[c] for c in rlhf_available])
        nonrlhf_delta = np.mean([inst_errs[c] - base_errs[c] for c in nonrlhf_available])

        return {
            'rlhf_delta': rlhf_delta,
            'nonrlhf_delta': nonrlhf_delta,
            'delta_gap': rlhf_delta - nonrlhf_delta
        }

    entanglement_sensitivity = run_seed_sensitivity(
        compute_entanglement_metrics,
        n_seeds=n_seeds,
        print_output=print_output
    )

    return {
        'base_probe': base_sensitivity,
        'instruct_probe': inst_sensitivity,
        'entanglement': entanglement_sensitivity
    }


def _get_category_errors_multi_seed(
    data: ModelData,
    position: str,
    n_seeds: int = 100,
    test_size: float = 0.2
) -> Dict[str, List[float]]:
    """
    Get probe error rates per category across multiple seeds.

    Helper function for significance testing.
    """
    X = get_activation_matrix(data, position)
    y = data.df['correct'].values.astype(int)
    categories = data.df['category'].values

    errors_by_cat = {cat: [] for cat in CATEGORIES}

    for seed in range(n_seeds):
        rs = 42 + seed

        X_train, X_test, y_train, y_test, _, idx_test = train_test_split(
            X, y, np.arange(len(y)), test_size=test_size, random_state=rs, stratify=y
        )

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=1000, random_state=rs)
        clf.fit(X_train_s, y_train)

        preds = clf.predict(X_test_s)
        errors = preds != y_test
        test_cats = categories[idx_test]

        for cat in CATEGORIES:
            mask = test_cats == cat
            if mask.sum() > 0:
                errors_by_cat[cat].append(errors[mask].mean())

    return errors_by_cat


def run_full_significance_analysis(
    base_data: ModelData,
    instruct_data: ModelData,
    position: str = 'last',
    n_permutations: int = 1000,
    n_seeds: int = 5,
    print_output: bool = True
) -> Dict:
    """
    Run complete significance analysis with all P0 requirements.

    Includes:
    1. Per-category significance tests with FDR correction
    2. Sample-level significance test (high power alternative to category-level)
    3. Seed sensitivity analysis

    Args:
        base_data: ModelData for base model
        instruct_data: ModelData for instruct model
        position: Token position
        n_permutations: Number of permutations
        n_seeds: Number of seeds for sensitivity
        print_output: Whether to print results

    Returns:
        Dictionary with all analysis results
    """
    if print_output:
        print("\n" + "=" * 80)
        print("COMPLETE SIGNIFICANCE ANALYSIS")
        print("=" * 80)
        print(f"Base: {base_data.name}")
        print(f"Instruct: {instruct_data.name}")

    results = {}

    # 1. Per-category significance testing with correction
    results['per_category'] = test_entanglement_significance(
        base_data, instruct_data, position,
        n_permutations=n_permutations,
        n_bootstrap=n_permutations,
        correction_method='fdr_bh',
        print_output=print_output
    )

    _cleanup()

    # 2. Sample-level significance test (high power)
    results['sample_level'] = test_entanglement_sample_level(
        base_data, instruct_data, position,
        n_permutations=n_permutations * 10,  # More permutations for main test
        n_cv_splits=min(n_permutations, 100),  # Cap CV splits for speed
        print_output=print_output
    )

    _cleanup()

    # 3. Seed sensitivity
    results['sensitivity'] = run_entanglement_seed_sensitivity(
        base_data, instruct_data, position,
        n_seeds=n_seeds,
        print_output=print_output
    )

    _cleanup()

    # 4. Summary
    if print_output:
        print("\n" + "=" * 80)
        print("ANALYSIS SUMMARY")
        print("=" * 80)

        # Sample-level result (main finding)
        sl = results['sample_level']
        print(f"\n1. SAMPLE-LEVEL TEST (n={sl['n_rlhf_samples']} vs {sl['n_nonrlhf_samples']}):")
        print(f"   RLHF error change:     {sl['rlhf_mean_change']:+.4f}")
        print(f"   Non-RLHF error change: {sl['nonrlhf_mean_change']:+.4f}")
        print(f"   Difference:            {sl['observed_difference']:+.4f}")
        print(f"   95% CI: [{sl['ci_low']:+.4f}, {sl['ci_high']:+.4f}]")
        print(f"   Cohen's d: {sl['cohens_d']:.3f}")
        print(f"   p-value: {sl['p_value']:.6f}")
        sig_str = "✓ SIGNIFICANT" if sl['significant'] else "✗ Not significant"
        print(f"   Result: {sig_str}")

        # Per-category corrected results
        pc = results['per_category']
        if pc['corrected_results']:
            cr = pc['corrected_results']
            print(f"\n2. PER-CATEGORY TESTS (FDR corrected):")
            print(f"   Significant after correction: {cr['n_significant']}/{cr['n_tests']}")

        # Category-level group comparison (note the limitation)
        if 'group_comparison' in pc and pc['group_comparison']:
            gc = pc['group_comparison']
            print(f"\n3. CATEGORY-LEVEL GROUP TEST (n=3 vs n=3, low power):")
            print(f"   Effect size: {gc['effect_size']:.3f}")
            print(f"   p-value: {gc['p_value']:.4f} (note: min possible is 0.05)")

        # Sensitivity
        sens = results['sensitivity']
        if 'entanglement' in sens:
            ent = sens['entanglement']
            print(f"\n4. SEED SENSITIVITY (CV < 5% = stable):")
            for metric, stable in ent.is_stable.items():
                status = "✓ stable" if stable else "⚠ unstable"
                cv = ent.summary[metric]['cv']
                print(f"   {metric}: CV={cv:.1%} ({status})")

    return results


# =============================================================================
# SAMPLE-LEVEL SIGNIFICANCE TEST (High Power Alternative)
# =============================================================================

def test_entanglement_sample_level(
    base_data: ModelData,
    instruct_data: ModelData,
    position: str = 'last',
    n_permutations: int = 10000,
    n_cv_splits: int = 100,
    print_output: bool = True
) -> Dict:
    """
    Sample-level significance test for RLHF vs non-RLHF entanglement.

    Instead of comparing 3 category means vs 3 category means (underpowered),
    this test compares ~300 RLHF samples vs ~300 non-RLHF samples directly.

    Method:
    1. For each sample, estimate probe error using repeated train/test splits
    2. Compute per-sample "error change" = instruct_error - base_error
    3. Compare error change distribution between RLHF and non-RLHF samples
    4. Use permutation test for significance (now with n >> 3)

    Args:
        base_data: ModelData for base model
        instruct_data: ModelData for instruct model
        position: Token position
        n_permutations: Number of permutations for test
        n_cv_splits: Number of CV splits to estimate per-sample error
        print_output: Whether to print results

    Returns:
        Dictionary with sample-level significance results
    """
    if print_output:
        print("\n" + "=" * 70)
        print("SAMPLE-LEVEL ENTANGLEMENT SIGNIFICANCE TEST")
        print("=" * 70)
        print(f"Base: {base_data.name} | Instruct: {instruct_data.name}")
        print(f"This test has much higher power than category-level (n >> 3)")

    # Get activations and labels
    X_base = get_activation_matrix(base_data, position)
    y_base = base_data.df['correct'].values.astype(int)
    cats_base = base_data.df['category'].values

    X_inst = get_activation_matrix(instruct_data, position)
    y_inst = instruct_data.df['correct'].values.astype(int)
    cats_inst = instruct_data.df['category'].values

    # Compute per-sample error estimates using repeated CV
    if print_output:
        print(f"\nEstimating per-sample probe errors ({n_cv_splits} CV splits)...")

    base_sample_errors = _get_sample_errors(X_base, y_base, n_splits=n_cv_splits)
    inst_sample_errors = _get_sample_errors(X_inst, y_inst, n_splits=n_cv_splits)

    # Compute per-sample error change (instruct - base)
    # Note: samples are in same order in both datasets
    error_change = inst_sample_errors - base_sample_errors

    # Assign samples to RLHF vs non-RLHF groups
    is_rlhf = np.array([cat in RLHF_CATEGORIES for cat in cats_base])
    is_nonrlhf = np.array([cat in NON_RLHF_CATEGORIES for cat in cats_base])

    rlhf_changes = error_change[is_rlhf]
    nonrlhf_changes = error_change[is_nonrlhf]

    n_rlhf = len(rlhf_changes)
    n_nonrlhf = len(nonrlhf_changes)

    if print_output:
        print(f"\nSample sizes:")
        print(f"  RLHF categories:     n = {n_rlhf}")
        print(f"  Non-RLHF categories: n = {n_nonrlhf}")

    # Observed effect
    observed_diff = rlhf_changes.mean() - nonrlhf_changes.mean()

    # Permutation test
    if print_output:
        print(f"\nRunning permutation test ({n_permutations} permutations)...")

    # Combine for permutation
    combined = np.concatenate([rlhf_changes, nonrlhf_changes])
    rng = np.random.RandomState(42)

    perm_diffs = np.zeros(n_permutations)
    for i in range(n_permutations):
        perm = rng.permutation(combined)
        perm_rlhf = perm[:n_rlhf]
        perm_nonrlhf = perm[n_rlhf:]
        perm_diffs[i] = perm_rlhf.mean() - perm_nonrlhf.mean()

    # One-sided p-value (RLHF expected to be higher)
    p_value = np.mean(perm_diffs >= observed_diff)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt(
        ((n_rlhf - 1) * rlhf_changes.var() + (n_nonrlhf - 1) * nonrlhf_changes.var())
        / (n_rlhf + n_nonrlhf - 2)
    )
    cohens_d = observed_diff / pooled_std if pooled_std > 0 else 0

    # Bootstrap CI for the difference
    boot_diffs = []
    for _ in range(2000):
        boot_rlhf = rng.choice(rlhf_changes, size=n_rlhf, replace=True)
        boot_nonrlhf = rng.choice(nonrlhf_changes, size=n_nonrlhf, replace=True)
        boot_diffs.append(boot_rlhf.mean() - boot_nonrlhf.mean())
    ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])

    results = {
        'n_rlhf_samples': n_rlhf,
        'n_nonrlhf_samples': n_nonrlhf,
        'rlhf_mean_change': float(rlhf_changes.mean()),
        'nonrlhf_mean_change': float(nonrlhf_changes.mean()),
        'observed_difference': float(observed_diff),
        'p_value': float(p_value),
        'cohens_d': float(cohens_d),
        'ci_low': float(ci_low),
        'ci_high': float(ci_high),
        'significant': p_value < 0.05,
        'n_permutations': n_permutations
    }

    if print_output:
        print(f"\n--- Results ---")
        print(f"RLHF mean error change:     {rlhf_changes.mean():+.4f}")
        print(f"Non-RLHF mean error change: {nonrlhf_changes.mean():+.4f}")
        print(f"Difference (RLHF - NonRLHF): {observed_diff:+.4f}")
        print(f"95% CI: [{ci_low:+.4f}, {ci_high:+.4f}]")
        print(f"Cohen's d: {cohens_d:.3f}")
        print(f"p-value (one-sided): {p_value:.6f}")

        if p_value < 0.001:
            sig_str = "*** HIGHLY SIGNIFICANT (p < 0.001)"
        elif p_value < 0.01:
            sig_str = "** SIGNIFICANT (p < 0.01)"
        elif p_value < 0.05:
            sig_str = "* SIGNIFICANT (p < 0.05)"
        else:
            sig_str = "Not significant"
        print(f"Result: {sig_str}")

        # Interpretation
        if results['significant']:
            print(f"\n✓ RLHF category samples show significantly greater probe error")
            print(f"  increase than non-RLHF samples after instruct tuning.")
            if cohens_d >= 0.8:
                print(f"  Effect size is LARGE (d = {cohens_d:.2f}).")
            elif cohens_d >= 0.5:
                print(f"  Effect size is MEDIUM (d = {cohens_d:.2f}).")
            else:
                print(f"  Effect size is SMALL (d = {cohens_d:.2f}).")

    return results


def _get_sample_errors(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 100,
    test_size: float = 0.2
) -> np.ndarray:
    """
    Estimate per-sample probe error using repeated train/test splits.

    For each sample, returns the average error rate when that sample
    was in the test set across all splits.

    Args:
        X: Feature matrix
        y: Labels
        n_splits: Number of train/test splits
        test_size: Fraction for test set

    Returns:
        Array of per-sample error estimates
    """
    n_samples = len(y)
    error_counts = np.zeros(n_samples)
    appear_counts = np.zeros(n_samples)

    for seed in range(n_splits):
        rs = 42 + seed

        # Split
        indices = np.arange(n_samples)
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, indices, test_size=test_size, random_state=rs, stratify=y
        )

        # Train probe
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=1000, random_state=rs)
        clf.fit(X_train_s, y_train)

        # Record errors for test samples
        preds = clf.predict(X_test_s)
        errors = (preds != y_test).astype(float)

        for i, idx in enumerate(idx_test):
            error_counts[idx] += errors[i]
            appear_counts[idx] += 1

    # Average error rate for each sample
    # Avoid division by zero for samples that never appeared in test set
    appear_counts = np.maximum(appear_counts, 1)
    sample_errors = error_counts / appear_counts

    return sample_errors
