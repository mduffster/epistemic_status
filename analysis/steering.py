"""
Steering vector analysis module.

D-STEER-inspired analysis to test if fine-tuning entanglement localizes
to a narrow subspace. Extracts steering vectors, projects activations,
and performs low-rank analysis via SVD.

References:
- D-STEER: Preference Alignment Techniques Learn to Behave, not to Believe
  (Gao et al. 2024, https://arxiv.org/abs/2512.11838)
"""

from typing import Dict, Optional, Tuple
import gc
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy import stats

from .loader import ModelData, get_activation_matrix


def _cleanup():
    """Force garbage collection."""
    gc.collect()


CATEGORIES = ['confident_correct', 'confident_incorrect', 'uncertain_correct',
              'uncertain_incorrect', 'ambiguous', 'nonsensical']

# Categories where fine-tuning trains specific epistemic behaviors
POLICY_CATEGORIES = ['confident_incorrect', 'nonsensical', 'ambiguous']
# Categories where correct response is knowledge recall
FACTUAL_CATEGORIES = ['confident_correct', 'uncertain_correct']


def _cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def _effect_size_label(d: float) -> str:
    """Return human-readable effect size label."""
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


def extract_steering_vector(
    base_data: ModelData,
    instruct_data: ModelData,
    position: str = 'last',
    layer: Optional[int] = None,
    normalize: bool = False,
    print_output: bool = True
) -> Dict:
    """
    Compute steering vector: mean(instruct) - mean(base).

    The steering vector captures the "direction of alignment" in representation
    space - the average displacement caused by fine-tuning.

    Args:
        base_data: ModelData for base model
        instruct_data: ModelData for instruct model
        position: Token position ('first', 'middle', 'last')
        layer: Specific layer or None for all layers flattened
        normalize: Whether to normalize the steering vector to unit length
        print_output: Whether to print results

    Returns:
        Dictionary with steering vector and metadata
    """
    X_base = get_activation_matrix(base_data, position, layer)
    X_instruct = get_activation_matrix(instruct_data, position, layer)

    # Compute centroids
    base_centroid = X_base.mean(axis=0)
    instruct_centroid = X_instruct.mean(axis=0)

    # Steering vector is the displacement
    steering_vector = instruct_centroid - base_centroid
    magnitude = np.linalg.norm(steering_vector)

    if normalize and magnitude > 0:
        steering_vector = steering_vector / magnitude

    results = {
        'steering_vector': steering_vector,
        'base_centroid': base_centroid,
        'instruct_centroid': instruct_centroid,
        'magnitude': float(magnitude),
        'layer': layer if layer is not None else 'all',
        'position': position,
        'n_dimensions': len(steering_vector),
        'normalized': normalize
    }

    if print_output:
        print("\n" + "=" * 60)
        print(f"STEERING VECTOR: {base_data.name} -> {instruct_data.name}")
        print("=" * 60)
        print(f"\nSteering Vector Properties:")
        print(f"  Position: {position} token")
        print(f"  Layer: {results['layer']}")
        print(f"  Dimensions: {results['n_dimensions']:,}")
        print(f"  Magnitude: {magnitude:.4f}")
        if normalize:
            print(f"  (normalized to unit length)")

    return results


def project_onto_steering(
    data: ModelData,
    steering_vector: np.ndarray,
    position: str = 'last',
    layer: Optional[int] = None,
    print_output: bool = True
) -> Dict:
    """
    Project each sample onto the steering direction.

    The projection tells us how much each sample aligns with the direction
    of fine-tuning. Differences between categories suggest localized effects.

    Args:
        data: ModelData object
        steering_vector: Steering vector from extract_steering_vector
        position: Token position
        layer: Specific layer or None for all
        print_output: Whether to print results

    Returns:
        Dictionary with projections by category
    """
    X = get_activation_matrix(data, position, layer)
    categories = data.df['category'].values

    # Normalize steering vector for projection
    steering_norm = np.linalg.norm(steering_vector)
    if steering_norm > 0:
        steering_unit = steering_vector / steering_norm
    else:
        steering_unit = steering_vector

    # Project: dot product with unit steering vector
    projections = X @ steering_unit

    results = {
        'projections': projections,
        'model': data.name,
        'by_category': {},
        'policy_vs_factual': {}
    }

    # Analyze by category
    for cat in CATEGORIES:
        mask = categories == cat
        if mask.sum() > 0:
            cat_projs = projections[mask]
            results['by_category'][cat] = {
                'mean': float(np.mean(cat_projs)),
                'std': float(np.std(cat_projs)),
                'min': float(np.min(cat_projs)),
                'max': float(np.max(cat_projs)),
                'n_samples': int(mask.sum())
            }

    # Policy vs Factual comparison
    policy_projs = projections[np.isin(categories, POLICY_CATEGORIES)]
    factual_projs = projections[np.isin(categories, FACTUAL_CATEGORIES)]

    if len(policy_projs) > 0 and len(factual_projs) > 0:
        policy_mean = float(np.mean(policy_projs))
        factual_mean = float(np.mean(factual_projs))
        difference = policy_mean - factual_mean
        effect_size = _cohens_d(policy_projs, factual_projs)

        results['policy_vs_factual'] = {
            'policy_mean': policy_mean,
            'factual_mean': factual_mean,
            'difference': difference,
            'effect_size': effect_size,
            'effect_label': _effect_size_label(effect_size)
        }

    if print_output:
        print(f"\nProjections onto Steering Direction ({data.name}):")
        print(f"{'Category':<25} {'Mean':>10} {'Std':>10} {'N':>6}")
        print("-" * 55)
        for cat in CATEGORIES:
            if cat in results['by_category']:
                r = results['by_category'][cat]
                print(f"{cat:<25} {r['mean']:>10.3f} {r['std']:>10.3f} {r['n_samples']:>6}")

        if results['policy_vs_factual']:
            pf = results['policy_vs_factual']
            print("-" * 55)
            print(f"Policy mean:  {pf['policy_mean']:>10.3f}")
            print(f"Factual mean: {pf['factual_mean']:>10.3f}")
            print(f"Difference:   {pf['difference']:>+10.3f} (d={pf['effect_size']:.2f}, {pf['effect_label']})")

    return results


def steering_by_category(
    base_data: ModelData,
    instruct_data: ModelData,
    position: str = 'last',
    layer: Optional[int] = None,
    print_output: bool = True
) -> Dict:
    """
    Full steering analysis: how much did each category move along the steering direction?

    This is the key test: if fine-tuning causes selective entanglement, policy
    categories should move further along the steering direction than factual ones.

    Args:
        base_data: ModelData for base model
        instruct_data: ModelData for instruct model
        position: Token position
        layer: Specific layer or None
        print_output: Whether to print results

    Returns:
        Dictionary with projection changes by category
    """
    # Extract steering vector
    sv_results = extract_steering_vector(
        base_data, instruct_data, position, layer,
        normalize=False, print_output=False
    )
    steering_vector = sv_results['steering_vector']

    # Project both base and instruct
    base_projs = project_onto_steering(
        base_data, steering_vector, position, layer, print_output=False
    )
    instruct_projs = project_onto_steering(
        instruct_data, steering_vector, position, layer, print_output=False
    )

    # Compute changes
    projection_change = {}
    base_cats = base_data.df['category'].values
    instruct_cats = instruct_data.df['category'].values

    for cat in CATEGORIES:
        if cat in base_projs['by_category'] and cat in instruct_projs['by_category']:
            base_mean = base_projs['by_category'][cat]['mean']
            instruct_mean = instruct_projs['by_category'][cat]['mean']

            projection_change[cat] = {
                'base_mean': base_mean,
                'instruct_mean': instruct_mean,
                'mean_change': instruct_mean - base_mean
            }

    # Aggregate by policy/factual
    policy_changes = [projection_change[c]['mean_change']
                      for c in POLICY_CATEGORIES if c in projection_change]
    factual_changes = [projection_change[c]['mean_change']
                       for c in FACTUAL_CATEGORIES if c in projection_change]

    policy_mean_change = np.mean(policy_changes) if policy_changes else 0
    factual_mean_change = np.mean(factual_changes) if factual_changes else 0

    # Effect size on the changes
    if len(policy_changes) > 1 and len(factual_changes) > 1:
        # Can't compute proper Cohen's d with n=3 per group, use simple ratio
        ratio = policy_mean_change / factual_mean_change if factual_mean_change != 0 else float('inf')
    else:
        ratio = None

    results = {
        'steering_vector': sv_results,
        'base_projections': base_projs,
        'instruct_projections': instruct_projs,
        'projection_change': projection_change,
        'summary': {
            'policy_mean_change': float(policy_mean_change),
            'factual_mean_change': float(factual_mean_change),
            'difference': float(policy_mean_change - factual_mean_change),
            'ratio': ratio
        },
        'interpretation': ''
    }

    # Generate interpretation
    if policy_mean_change > factual_mean_change:
        if ratio and ratio > 2:
            results['interpretation'] = (
                f"Policy categories moved {ratio:.1f}x further along the steering direction "
                f"than factual categories, consistent with D-STEER's 'narrow subspace' hypothesis. "
                f"Entanglement appears concentrated in the alignment direction."
            )
        else:
            results['interpretation'] = (
                f"Policy categories moved further along the steering direction than factual "
                f"categories (diff={policy_mean_change - factual_mean_change:.3f}), suggesting "
                f"some localization of fine-tuning effects."
            )
    else:
        results['interpretation'] = (
            f"Policy and factual categories moved similarly along the steering direction. "
            f"Entanglement may not localize to the mean steering vector."
        )

    if print_output:
        print("\n" + "=" * 60)
        print(f"STEERING ANALYSIS: {base_data.name} -> {instruct_data.name}")
        print("=" * 60)

        print(f"\nSteering vector magnitude: {sv_results['magnitude']:.4f}")

        print(f"\nCategory Projections onto Steering Direction:")
        print(f"{'Category':<25} {'Base':>10} {'Instruct':>10} {'Change':>10}")
        print("-" * 60)

        for cat in CATEGORIES:
            if cat in projection_change:
                pc = projection_change[cat]
                marker = "***" if cat in POLICY_CATEGORIES else ""
                print(f"{cat:<25} {pc['base_mean']:>10.3f} {pc['instruct_mean']:>10.3f} "
                      f"{pc['mean_change']:>+10.3f} {marker}")

        print("-" * 60)
        print(f"{'Policy mean change:':<25} {policy_mean_change:>+31.3f}")
        print(f"{'Factual mean change:':<25} {factual_mean_change:>+31.3f}")
        print(f"{'Difference:':<25} {policy_mean_change - factual_mean_change:>+31.3f}")
        if ratio:
            print(f"{'Ratio (policy/factual):':<25} {ratio:>31.1f}x")

        print(f"\nINTERPRETATION: {results['interpretation']}")

    return results


def low_rank_analysis(
    base_data: ModelData,
    instruct_data: ModelData,
    position: str = 'last',
    layer: Optional[int] = None,
    n_components: int = 50,
    variance_threshold: float = 0.80,
    print_output: bool = True
) -> Dict:
    """
    SVD analysis of alignment changes to test low-rank hypothesis.

    If fine-tuning operates in a narrow subspace (per D-STEER), the difference
    matrix should be low-rank: a small number of singular values capture most
    variance.

    Math:
        D = instruct_activations - base_activations
        D = U @ Sigma @ V^T

        - Sigma: singular values (importance of each component)
        - V^T: directions in activation space
        - U @ Sigma: how each sample loads on components

    Args:
        base_data: ModelData for base model
        instruct_data: ModelData for instruct model
        position: Token position
        layer: Specific layer or None
        n_components: Number of components to analyze
        variance_threshold: Threshold for "effective rank"
        print_output: Whether to print results

    Returns:
        Dictionary with SVD results and category loadings
    """
    X_base = get_activation_matrix(base_data, position, layer)
    X_instruct = get_activation_matrix(instruct_data, position, layer)

    # Difference matrix: each row is how that sample changed
    D = X_instruct - X_base

    # Center the difference matrix
    D_centered = D - D.mean(axis=0)

    # SVD (truncated for efficiency)
    n_components = min(n_components, min(D.shape) - 1)

    # Full SVD on centered difference
    U, S, Vt = np.linalg.svd(D_centered, full_matrices=False)

    # Keep only top n_components
    U = U[:, :n_components]
    S = S[:n_components]
    Vt = Vt[:n_components, :]

    # Variance explained
    total_variance = np.sum(S ** 2)
    variance_explained = np.cumsum(S ** 2) / total_variance

    # Effective rank: components needed for threshold variance
    effective_rank = int(np.searchsorted(variance_explained, variance_threshold) + 1)

    # Sample loadings: U @ diag(S) shows how each sample projects onto components
    sample_loadings = U * S  # Broadcasting: (n_samples, n_components)

    # Analyze loadings by category
    categories = base_data.df['category'].values
    by_category = {}

    for cat in CATEGORIES:
        mask = categories == cat
        if mask.sum() > 0:
            cat_loadings = sample_loadings[mask]
            mean_loading = cat_loadings.mean(axis=0)

            # Loading magnitude on top-k components (how much this category uses them)
            top_k = min(10, n_components)
            loading_magnitude = np.linalg.norm(mean_loading[:top_k])

            by_category[cat] = {
                'mean_loading': mean_loading[:top_k].tolist(),
                'loading_magnitude': float(loading_magnitude),
                'n_samples': int(mask.sum())
            }

    # Compare policy vs factual loading magnitudes
    policy_mags = [by_category[c]['loading_magnitude']
                   for c in POLICY_CATEGORIES if c in by_category]
    factual_mags = [by_category[c]['loading_magnitude']
                    for c in FACTUAL_CATEGORIES if c in by_category]

    policy_mean_mag = np.mean(policy_mags) if policy_mags else 0
    factual_mean_mag = np.mean(factual_mags) if factual_mags else 0

    results = {
        'singular_values': S.tolist(),
        'variance_explained': variance_explained.tolist(),
        'effective_rank': effective_rank,
        'variance_threshold': variance_threshold,
        'total_variance': float(total_variance),
        'n_components_analyzed': n_components,
        'top_directions': Vt[:10].tolist() if len(Vt) >= 10 else Vt.tolist(),
        'by_category': by_category,
        'summary': {
            'policy_mean_loading_magnitude': float(policy_mean_mag),
            'factual_mean_loading_magnitude': float(factual_mean_mag),
            'ratio': float(policy_mean_mag / factual_mean_mag) if factual_mean_mag > 0 else None
        },
        'interpretation': ''
    }

    # Generate interpretation
    if effective_rank <= 20:
        rank_interp = f"low-rank ({effective_rank} dimensions capture {variance_threshold*100:.0f}% variance)"
    else:
        rank_interp = f"distributed ({effective_rank} dimensions for {variance_threshold*100:.0f}% variance)"

    if policy_mean_mag > factual_mean_mag * 1.5:
        loading_interp = (
            f"Policy categories load {policy_mean_mag/factual_mean_mag:.1f}x more heavily "
            f"on top components than factual categories."
        )
    else:
        loading_interp = "Policy and factual categories load similarly on top components."

    results['interpretation'] = f"Alignment changes are {rank_interp}. {loading_interp}"

    if print_output:
        print("\n" + "=" * 60)
        print("LOW-RANK ANALYSIS (SVD)")
        print("=" * 60)

        print("\nSingular Value Spectrum:")
        cumulative = 0
        for i in range(min(10, n_components)):
            var_i = (S[i] ** 2) / total_variance * 100
            cumulative += var_i
            print(f"  Component {i+1:2d}: {var_i:5.1f}% variance  (cumulative: {cumulative:5.1f}%)")
        if n_components > 10:
            var_20 = variance_explained[min(19, n_components-1)] * 100
            print(f"  ...")
            print(f"  Component 20: {var_20:5.1f}% cumulative")

        print(f"\nEffective rank ({variance_threshold*100:.0f}% variance): {effective_rank} components")

        print(f"\nCategory Loadings on Top-10 Components:")
        print(f"{'Category':<25} {'Loading Mag':>15}")
        print("-" * 45)

        for cat in CATEGORIES:
            if cat in by_category:
                mag = by_category[cat]['loading_magnitude']
                marker = "***" if cat in POLICY_CATEGORIES and mag > factual_mean_mag * 1.5 else ""
                print(f"{cat:<25} {mag:>15.3f} {marker}")

        print("-" * 45)
        print(f"Policy mean:  {policy_mean_mag:>15.3f}")
        print(f"Factual mean: {factual_mean_mag:>15.3f}")
        if results['summary']['ratio']:
            print(f"Ratio:        {results['summary']['ratio']:>15.1f}x")

        print(f"\nINTERPRETATION: {results['interpretation']}")

    _cleanup()
    return results


def ablate_steering_subspace(
    base_data: ModelData,
    instruct_data: ModelData,
    n_components: int = 10,
    position: str = 'last',
    layer: Optional[int] = None,
    print_output: bool = True
) -> Dict:
    """
    Remove top-k SVD components and measure impact on probe performance.

    This is a causal test: if entanglement is localized to the steering
    subspace, removing those dimensions should reduce the policy/factual
    gap in probe error rates.

    Args:
        base_data: ModelData for base model
        instruct_data: ModelData for instruct model
        n_components: Number of components to ablate
        position: Token position
        layer: Specific layer or None
        print_output: Whether to print results

    Returns:
        Dictionary with ablation results
    """
    X_base = get_activation_matrix(base_data, position, layer)
    X_instruct = get_activation_matrix(instruct_data, position, layer)

    # Get difference matrix and SVD
    D = X_instruct - X_base
    D_centered = D - D.mean(axis=0)

    U, S, Vt = np.linalg.svd(D_centered, full_matrices=False)

    # Top-k directions to ablate
    top_k_directions = Vt[:n_components, :]  # (k, hidden_dim)

    # Project out top-k directions from instruct activations
    # X_ablated = X - sum_i (X @ v_i) * v_i^T
    X_instruct_ablated = X_instruct.copy()
    for i in range(n_components):
        v_i = top_k_directions[i]
        projection = (X_instruct_ablated @ v_i).reshape(-1, 1)
        X_instruct_ablated = X_instruct_ablated - projection @ v_i.reshape(1, -1)

    # Train probes on original and ablated
    y_base = base_data.df['correct'].values.astype(int)
    y_instruct = instruct_data.df['correct'].values.astype(int)
    categories_instruct = instruct_data.df['category'].values

    def probe_error_by_category(X, y, categories):
        """Train probe and get error rate by category."""
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, np.arange(len(y)), test_size=0.2, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train_scaled, y_train)

        preds = clf.predict(X_test_scaled)
        errors = preds != y_test
        test_cats = categories[idx_test]

        results = {}
        for cat in CATEGORIES:
            mask = test_cats == cat
            if mask.sum() > 0:
                results[cat] = float(errors[mask].mean())
        return results

    # Original instruct errors
    original_errors = probe_error_by_category(X_instruct, y_instruct, categories_instruct)

    # Ablated instruct errors
    ablated_errors = probe_error_by_category(X_instruct_ablated, y_instruct, categories_instruct)

    # Compute changes
    error_change = {}
    for cat in CATEGORIES:
        if cat in original_errors and cat in ablated_errors:
            error_change[cat] = {
                'original': original_errors[cat],
                'ablated': ablated_errors[cat],
                'change': ablated_errors[cat] - original_errors[cat]
            }

    # Aggregate
    policy_orig = np.mean([original_errors[c] for c in POLICY_CATEGORIES if c in original_errors])
    factual_orig = np.mean([original_errors[c] for c in FACTUAL_CATEGORIES if c in original_errors])
    policy_abl = np.mean([ablated_errors[c] for c in POLICY_CATEGORIES if c in ablated_errors])
    factual_abl = np.mean([ablated_errors[c] for c in FACTUAL_CATEGORIES if c in ablated_errors])

    original_gap = policy_orig - factual_orig
    ablated_gap = policy_abl - factual_abl
    gap_reduction = original_gap - ablated_gap

    results = {
        'n_components_ablated': n_components,
        'original_errors': original_errors,
        'ablated_errors': ablated_errors,
        'error_change': error_change,
        'summary': {
            'original_policy_error': float(policy_orig),
            'original_factual_error': float(factual_orig),
            'original_gap': float(original_gap),
            'ablated_policy_error': float(policy_abl),
            'ablated_factual_error': float(factual_abl),
            'ablated_gap': float(ablated_gap),
            'gap_reduction': float(gap_reduction),
            'gap_reduction_pct': float(gap_reduction / original_gap * 100) if original_gap != 0 else 0
        },
        'interpretation': ''
    }

    # Generate interpretation
    if gap_reduction > 0.05:
        results['interpretation'] = (
            f"Ablating top-{n_components} steering components reduced the policy-factual "
            f"error gap by {gap_reduction:.3f} ({results['summary']['gap_reduction_pct']:.0f}%). "
            f"This suggests entanglement is concentrated in the steering subspace."
        )
    elif gap_reduction > 0:
        results['interpretation'] = (
            f"Ablating top-{n_components} steering components slightly reduced the gap "
            f"({gap_reduction:.3f}). Entanglement may be partially localized."
        )
    else:
        results['interpretation'] = (
            f"Ablating steering components did not reduce the policy-factual gap. "
            f"Entanglement may be distributed across the full representation space."
        )

    if print_output:
        print("\n" + "=" * 60)
        print(f"ABLATION ANALYSIS (removing top-{n_components} components)")
        print("=" * 60)

        print(f"\nProbe Error Rates by Category:")
        print(f"{'Category':<25} {'Original':>10} {'Ablated':>10} {'Change':>10}")
        print("-" * 60)

        for cat in CATEGORIES:
            if cat in error_change:
                ec = error_change[cat]
                print(f"{cat:<25} {ec['original']:>10.3f} {ec['ablated']:>10.3f} "
                      f"{ec['change']:>+10.3f}")

        print("-" * 60)
        s = results['summary']
        print(f"{'Policy mean:':<25} {s['original_policy_error']:>10.3f} "
              f"{s['ablated_policy_error']:>10.3f}")
        print(f"{'Factual mean:':<25} {s['original_factual_error']:>10.3f} "
              f"{s['ablated_factual_error']:>10.3f}")
        print(f"{'Gap (policy-factual):':<25} {s['original_gap']:>10.3f} "
              f"{s['ablated_gap']:>10.3f}")
        print(f"\nGap reduction: {s['gap_reduction']:+.3f} ({s['gap_reduction_pct']:.0f}%)")

        print(f"\nINTERPRETATION: {results['interpretation']}")

    _cleanup()
    return results


def run_full_steering_analysis(
    base_data: ModelData,
    instruct_data: ModelData,
    position: str = 'last',
    n_ablate_components: int = 10,
    print_output: bool = True
) -> Dict:
    """
    Run complete steering vector analysis suite.

    Combines:
    1. Steering vector extraction
    2. Category-wise projection analysis
    3. Low-rank SVD analysis
    4. Ablation test

    Args:
        base_data: ModelData for base model
        instruct_data: ModelData for instruct model
        position: Token position
        n_ablate_components: Components to ablate in causal test
        print_output: Whether to print results

    Returns:
        Combined results dictionary
    """
    if print_output:
        print("\n" + "=" * 70)
        print(f"FULL STEERING ANALYSIS: {base_data.name} -> {instruct_data.name}")
        print("=" * 70)

    # 1. Steering by category
    steering_results = steering_by_category(
        base_data, instruct_data, position, print_output=print_output
    )

    # 2. Low-rank analysis
    lowrank_results = low_rank_analysis(
        base_data, instruct_data, position, print_output=print_output
    )

    # 3. Ablation test
    ablation_results = ablate_steering_subspace(
        base_data, instruct_data, n_components=n_ablate_components,
        position=position, print_output=print_output
    )

    # Combined results
    results = {
        'base_model': base_data.name,
        'instruct_model': instruct_data.name,
        'position': position,
        'steering': steering_results,
        'low_rank': lowrank_results,
        'ablation': ablation_results,
        'overall_interpretation': ''
    }

    # Generate overall interpretation
    interps = []

    # Steering direction finding
    steering_ratio = steering_results['summary'].get('ratio')
    if steering_ratio and steering_ratio > 2:
        interps.append(f"Policy categories moved {steering_ratio:.1f}x further along steering direction")

    # Low-rank finding
    effective_rank = lowrank_results['effective_rank']
    if effective_rank <= 20:
        interps.append(f"alignment is low-rank ({effective_rank} dimensions)")

    # Ablation finding
    gap_reduction_pct = ablation_results['summary']['gap_reduction_pct']
    if gap_reduction_pct > 20:
        interps.append(f"ablating top components reduced gap by {gap_reduction_pct:.0f}%")

    if interps:
        results['overall_interpretation'] = (
            f"Evidence for localized entanglement: {'; '.join(interps)}. "
            f"Consistent with D-STEER's narrow subspace hypothesis."
        )
    else:
        results['overall_interpretation'] = (
            "Mixed evidence for localized entanglement. Effects may be distributed."
        )

    if print_output:
        print("\n" + "=" * 70)
        print("OVERALL SUMMARY")
        print("=" * 70)
        print(f"\n{results['overall_interpretation']}")

    return results


# =============================================================================
# CATEGORY-SPECIFIC STEERING ANALYSIS
# =============================================================================

def extract_category_steering_vectors(
    base_data: ModelData,
    instruct_data: ModelData,
    position: str = 'last',
    layer: Optional[int] = None,
    print_output: bool = True
) -> Dict:
    """
    Compute separate steering vectors for policy and factual categories.

    Instead of a single mean steering vector, this computes:
    - policy_steering: mean change for policy categories only
    - factual_steering: mean change for factual categories only
    - differential_steering: policy_steering - factual_steering

    The differential captures the direction where policy and factual diverge.

    Args:
        base_data: ModelData for base model
        instruct_data: ModelData for instruct model
        position: Token position
        layer: Specific layer or None
        print_output: Whether to print results

    Returns:
        Dictionary with category-specific steering vectors
    """
    X_base = get_activation_matrix(base_data, position, layer)
    X_instruct = get_activation_matrix(instruct_data, position, layer)

    base_cats = base_data.df['category'].values
    instruct_cats = instruct_data.df['category'].values

    # Policy steering: mean(instruct[policy]) - mean(base[policy])
    policy_mask_base = np.isin(base_cats, POLICY_CATEGORIES)
    policy_mask_inst = np.isin(instruct_cats, POLICY_CATEGORIES)

    policy_base_centroid = X_base[policy_mask_base].mean(axis=0)
    policy_inst_centroid = X_instruct[policy_mask_inst].mean(axis=0)
    policy_steering = policy_inst_centroid - policy_base_centroid

    # Factual steering: mean(instruct[factual]) - mean(base[factual])
    factual_mask_base = np.isin(base_cats, FACTUAL_CATEGORIES)
    factual_mask_inst = np.isin(instruct_cats, FACTUAL_CATEGORIES)

    factual_base_centroid = X_base[factual_mask_base].mean(axis=0)
    factual_inst_centroid = X_instruct[factual_mask_inst].mean(axis=0)
    factual_steering = factual_inst_centroid - factual_base_centroid

    # Differential: where policy and factual diverge
    differential_steering = policy_steering - factual_steering

    # Overall steering for comparison
    overall_steering = X_instruct.mean(axis=0) - X_base.mean(axis=0)

    # Compute magnitudes
    policy_mag = np.linalg.norm(policy_steering)
    factual_mag = np.linalg.norm(factual_steering)
    differential_mag = np.linalg.norm(differential_steering)
    overall_mag = np.linalg.norm(overall_steering)

    # Compute angles between vectors (cosine similarity)
    def cosine_sim(v1, v2):
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    policy_factual_sim = cosine_sim(policy_steering, factual_steering)
    policy_overall_sim = cosine_sim(policy_steering, overall_steering)
    factual_overall_sim = cosine_sim(factual_steering, overall_steering)
    differential_overall_sim = cosine_sim(differential_steering, overall_steering)

    results = {
        'policy_steering': policy_steering,
        'factual_steering': factual_steering,
        'differential_steering': differential_steering,
        'overall_steering': overall_steering,
        'magnitudes': {
            'policy': float(policy_mag),
            'factual': float(factual_mag),
            'differential': float(differential_mag),
            'overall': float(overall_mag)
        },
        'similarities': {
            'policy_factual': policy_factual_sim,
            'policy_overall': policy_overall_sim,
            'factual_overall': factual_overall_sim,
            'differential_overall': differential_overall_sim
        },
        'n_policy_samples': int(policy_mask_base.sum()),
        'n_factual_samples': int(factual_mask_base.sum())
    }

    if print_output:
        print("\n" + "=" * 60)
        print("CATEGORY-SPECIFIC STEERING VECTORS")
        print("=" * 60)

        print("\nSteering Vector Magnitudes:")
        print(f"  Policy steering:      {policy_mag:>10.3f}")
        print(f"  Factual steering:     {factual_mag:>10.3f}")
        print(f"  Differential:         {differential_mag:>10.3f}")
        print(f"  Overall (mean):       {overall_mag:>10.3f}")

        print("\nCosine Similarities:")
        print(f"  Policy ↔ Factual:     {policy_factual_sim:>10.3f}")
        print(f"  Policy ↔ Overall:     {policy_overall_sim:>10.3f}")
        print(f"  Factual ↔ Overall:    {factual_overall_sim:>10.3f}")
        print(f"  Differential ↔ Overall: {differential_overall_sim:>10.3f}")

        if policy_factual_sim < 0.9:
            print(f"\n  Policy and factual steering directions diverge "
                  f"(sim={policy_factual_sim:.2f} < 0.9)")

    return results


def category_specific_projection_analysis(
    base_data: ModelData,
    instruct_data: ModelData,
    position: str = 'last',
    layer: Optional[int] = None,
    print_output: bool = True
) -> Dict:
    """
    Project samples onto category-specific steering directions.

    Key test: If entanglement is category-specific, then:
    - Policy samples should project more onto policy_steering than factual_steering
    - The differential_steering should separate policy from factual categories

    Args:
        base_data: ModelData for base model
        instruct_data: ModelData for instruct model
        position: Token position
        layer: Specific layer or None
        print_output: Whether to print results

    Returns:
        Dictionary with projection analysis
    """
    # Get steering vectors
    sv = extract_category_steering_vectors(
        base_data, instruct_data, position, layer, print_output=False
    )

    X_base = get_activation_matrix(base_data, position, layer)
    X_instruct = get_activation_matrix(instruct_data, position, layer)
    base_cats = base_data.df['category'].values
    instruct_cats = instruct_data.df['category'].values

    # Normalize steering vectors
    def normalize(v):
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    policy_unit = normalize(sv['policy_steering'])
    factual_unit = normalize(sv['factual_steering'])
    differential_unit = normalize(sv['differential_steering'])

    # Compute change vectors for each sample
    D = X_instruct - X_base  # (n_samples, hidden_dim)

    # Project changes onto each steering direction
    proj_policy = D @ policy_unit
    proj_factual = D @ factual_unit
    proj_differential = D @ differential_unit

    # Analyze by category
    by_category = {}
    for cat in CATEGORIES:
        mask = base_cats == cat
        if mask.sum() > 0:
            by_category[cat] = {
                'proj_policy': float(proj_policy[mask].mean()),
                'proj_factual': float(proj_factual[mask].mean()),
                'proj_differential': float(proj_differential[mask].mean()),
                'proj_policy_std': float(proj_policy[mask].std()),
                'proj_differential_std': float(proj_differential[mask].std()),
                'n_samples': int(mask.sum())
            }

    # Summary: policy vs factual categories
    policy_mask = np.isin(base_cats, POLICY_CATEGORIES)
    factual_mask = np.isin(base_cats, FACTUAL_CATEGORIES)

    # Key metric: how do policy vs factual samples project onto the differential?
    policy_on_diff = proj_differential[policy_mask]
    factual_on_diff = proj_differential[factual_mask]

    diff_separation = float(policy_on_diff.mean() - factual_on_diff.mean())
    diff_effect_size = _cohens_d(policy_on_diff, factual_on_diff)

    results = {
        'steering_vectors': sv,
        'by_category': by_category,
        'summary': {
            'policy_mean_on_differential': float(policy_on_diff.mean()),
            'factual_mean_on_differential': float(factual_on_diff.mean()),
            'differential_separation': diff_separation,
            'differential_effect_size': diff_effect_size,
            'effect_label': _effect_size_label(diff_effect_size)
        },
        'interpretation': ''
    }

    # Generate interpretation
    if abs(diff_effect_size) > 0.8:
        results['interpretation'] = (
            f"Strong separation (d={diff_effect_size:.2f}): Policy and factual categories "
            f"move in different directions during fine-tuning. The differential steering "
            f"vector captures the entanglement direction."
        )
    elif abs(diff_effect_size) > 0.5:
        results['interpretation'] = (
            f"Moderate separation (d={diff_effect_size:.2f}): Some divergence between "
            f"policy and factual category changes."
        )
    else:
        results['interpretation'] = (
            f"Weak separation (d={diff_effect_size:.2f}): Policy and factual categories "
            f"change similarly during fine-tuning."
        )

    if print_output:
        print("\n" + "=" * 60)
        print("CATEGORY-SPECIFIC PROJECTION ANALYSIS")
        print("=" * 60)

        print("\nProjection of Changes onto Steering Directions:")
        print(f"{'Category':<20} {'→Policy':>10} {'→Factual':>10} {'→Diff':>10}")
        print("-" * 55)

        for cat in CATEGORIES:
            if cat in by_category:
                bc = by_category[cat]
                marker = "**" if cat in POLICY_CATEGORIES else ""
                print(f"{cat:<20} {bc['proj_policy']:>10.2f} {bc['proj_factual']:>10.2f} "
                      f"{bc['proj_differential']:>10.2f} {marker}")

        print("-" * 55)
        s = results['summary']
        print(f"\nDifferential Steering Separation:")
        print(f"  Policy categories mean:  {s['policy_mean_on_differential']:>+10.2f}")
        print(f"  Factual categories mean: {s['factual_mean_on_differential']:>+10.2f}")
        print(f"  Separation:              {s['differential_separation']:>+10.2f}")
        print(f"  Effect size:             {s['differential_effect_size']:>10.2f} ({s['effect_label']})")

        print(f"\nINTERPRETATION: {results['interpretation']}")

    return results


def ablate_differential_steering(
    base_data: ModelData,
    instruct_data: ModelData,
    position: str = 'last',
    layer: Optional[int] = None,
    print_output: bool = True
) -> Dict:
    """
    Ablate the differential steering direction and re-run probe analysis.

    If entanglement is captured by the differential (policy - factual) steering
    direction, removing it should reduce the policy-factual probe error gap.

    Args:
        base_data: ModelData for base model
        instruct_data: ModelData for instruct model
        position: Token position
        layer: Specific layer or None
        print_output: Whether to print results

    Returns:
        Dictionary with ablation results
    """
    # Get steering vectors
    sv = extract_category_steering_vectors(
        base_data, instruct_data, position, layer, print_output=False
    )

    X_instruct = get_activation_matrix(instruct_data, position, layer)
    y_instruct = instruct_data.df['correct'].values.astype(int)
    categories_instruct = instruct_data.df['category'].values

    # Normalize differential steering
    diff_steering = sv['differential_steering']
    diff_norm = np.linalg.norm(diff_steering)
    if diff_norm > 0:
        diff_unit = diff_steering / diff_norm
    else:
        diff_unit = diff_steering

    # Ablate: remove projection onto differential steering
    projections = (X_instruct @ diff_unit).reshape(-1, 1)
    X_instruct_ablated = X_instruct - projections @ diff_unit.reshape(1, -1)

    def probe_error_by_category(X, y, categories):
        """Train probe and get error rate by category."""
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, np.arange(len(y)), test_size=0.2, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train_scaled, y_train)

        preds = clf.predict(X_test_scaled)
        errors = preds != y_test
        test_cats = categories[idx_test]

        results = {}
        for cat in CATEGORIES:
            mask = test_cats == cat
            if mask.sum() > 0:
                results[cat] = float(errors[mask].mean())
        return results

    # Original errors
    original_errors = probe_error_by_category(X_instruct, y_instruct, categories_instruct)

    # Ablated errors
    ablated_errors = probe_error_by_category(X_instruct_ablated, y_instruct, categories_instruct)

    # Compute changes
    error_change = {}
    for cat in CATEGORIES:
        if cat in original_errors and cat in ablated_errors:
            error_change[cat] = {
                'original': original_errors[cat],
                'ablated': ablated_errors[cat],
                'change': ablated_errors[cat] - original_errors[cat]
            }

    # Aggregate
    policy_orig = np.mean([original_errors[c] for c in POLICY_CATEGORIES if c in original_errors])
    factual_orig = np.mean([original_errors[c] for c in FACTUAL_CATEGORIES if c in original_errors])
    policy_abl = np.mean([ablated_errors[c] for c in POLICY_CATEGORIES if c in ablated_errors])
    factual_abl = np.mean([ablated_errors[c] for c in FACTUAL_CATEGORIES if c in ablated_errors])

    original_gap = policy_orig - factual_orig
    ablated_gap = policy_abl - factual_abl
    gap_reduction = original_gap - ablated_gap

    results = {
        'differential_magnitude': float(diff_norm),
        'original_errors': original_errors,
        'ablated_errors': ablated_errors,
        'error_change': error_change,
        'summary': {
            'original_policy_error': float(policy_orig),
            'original_factual_error': float(factual_orig),
            'original_gap': float(original_gap),
            'ablated_policy_error': float(policy_abl),
            'ablated_factual_error': float(factual_abl),
            'ablated_gap': float(ablated_gap),
            'gap_reduction': float(gap_reduction),
            'gap_reduction_pct': float(gap_reduction / original_gap * 100) if original_gap != 0 else 0
        },
        'interpretation': ''
    }

    # Generate interpretation
    if gap_reduction > 0.05:
        results['interpretation'] = (
            f"Ablating the differential steering direction reduced the policy-factual "
            f"gap by {gap_reduction:.3f} ({results['summary']['gap_reduction_pct']:.0f}%). "
            f"Entanglement IS captured by the policy-factual divergence direction."
        )
    elif gap_reduction > 0:
        results['interpretation'] = (
            f"Slight gap reduction ({gap_reduction:.3f}). Entanglement partially captured."
        )
    else:
        results['interpretation'] = (
            f"Ablating differential steering did not reduce the gap. "
            f"Entanglement may be orthogonal to the policy-factual divergence."
        )

    if print_output:
        print("\n" + "=" * 60)
        print("DIFFERENTIAL STEERING ABLATION")
        print("=" * 60)

        print(f"\nDifferential steering magnitude: {diff_norm:.3f}")

        print(f"\nProbe Error Rates:")
        print(f"{'Category':<25} {'Original':>10} {'Ablated':>10} {'Change':>10}")
        print("-" * 60)

        for cat in CATEGORIES:
            if cat in error_change:
                ec = error_change[cat]
                print(f"{cat:<25} {ec['original']:>10.3f} {ec['ablated']:>10.3f} "
                      f"{ec['change']:>+10.3f}")

        print("-" * 60)
        s = results['summary']
        print(f"{'Policy mean:':<25} {s['original_policy_error']:>10.3f} "
              f"{s['ablated_policy_error']:>10.3f}")
        print(f"{'Factual mean:':<25} {s['original_factual_error']:>10.3f} "
              f"{s['ablated_factual_error']:>10.3f}")
        print(f"{'Gap:':<25} {s['original_gap']:>10.3f} {s['ablated_gap']:>10.3f}")
        print(f"\nGap reduction: {s['gap_reduction']:+.3f} ({s['gap_reduction_pct']:.0f}%)")

        print(f"\nINTERPRETATION: {results['interpretation']}")

    return results


def run_category_specific_steering_analysis(
    base_data: ModelData,
    instruct_data: ModelData,
    position: str = 'last',
    print_output: bool = True
) -> Dict:
    """
    Run complete category-specific steering analysis.

    This analysis tests whether entanglement localizes to the direction
    where policy and factual categories diverge during fine-tuning.

    Args:
        base_data: ModelData for base model
        instruct_data: ModelData for instruct model
        position: Token position
        print_output: Whether to print results

    Returns:
        Combined results dictionary
    """
    if print_output:
        print("\n" + "=" * 70)
        print(f"CATEGORY-SPECIFIC STEERING: {base_data.name} -> {instruct_data.name}")
        print("=" * 70)

    # 1. Extract category-specific steering vectors
    sv_results = extract_category_steering_vectors(
        base_data, instruct_data, position, print_output=print_output
    )

    # 2. Projection analysis
    proj_results = category_specific_projection_analysis(
        base_data, instruct_data, position, print_output=print_output
    )

    # 3. Differential ablation
    ablation_results = ablate_differential_steering(
        base_data, instruct_data, position, print_output=print_output
    )

    results = {
        'base_model': base_data.name,
        'instruct_model': instruct_data.name,
        'position': position,
        'steering_vectors': {
            'magnitudes': sv_results['magnitudes'],
            'similarities': sv_results['similarities']
        },
        'projection_analysis': {
            'by_category': proj_results['by_category'],
            'summary': proj_results['summary']
        },
        'ablation': ablation_results,
        'overall_interpretation': ''
    }

    # Generate overall interpretation
    diff_effect = proj_results['summary']['differential_effect_size']
    gap_reduction = ablation_results['summary']['gap_reduction_pct']
    policy_factual_sim = sv_results['similarities']['policy_factual']

    findings = []

    if policy_factual_sim < 0.9:
        findings.append(f"policy/factual steering directions diverge (sim={policy_factual_sim:.2f})")

    if abs(diff_effect) > 0.5:
        findings.append(f"differential separates categories (d={diff_effect:.2f})")

    if gap_reduction > 10:
        findings.append(f"ablation reduced gap by {gap_reduction:.0f}%")

    if findings:
        results['overall_interpretation'] = (
            f"Evidence for category-specific entanglement: {'; '.join(findings)}. "
            f"Fine-tuning affects policy and factual categories differently."
        )
    else:
        results['overall_interpretation'] = (
            "Policy and factual categories change similarly during fine-tuning. "
            "Entanglement may not be category-specific in direction."
        )

    if print_output:
        print("\n" + "=" * 70)
        print("OVERALL SUMMARY")
        print("=" * 70)
        print(f"\n{results['overall_interpretation']}")

    _cleanup()
    return results


# =============================================================================
# SUBCATEGORY CONVERGENCE ANALYSIS
# =============================================================================

def extract_subcategory_steering_vectors(
    base_data: ModelData,
    instruct_data: ModelData,
    position: str = 'last',
    layer: Optional[int] = None,
    print_output: bool = True
) -> Dict:
    """
    Compute steering vectors for each individual category.

    Instead of aggregating into policy/factual, compute:
    - steering_i = mean(instruct[cat_i]) - mean(base[cat_i])

    This allows us to test whether specific categories converge during fine-tuning.

    Args:
        base_data: ModelData for base model
        instruct_data: ModelData for instruct model
        position: Token position
        layer: Specific layer or None
        print_output: Whether to print results

    Returns:
        Dictionary with per-category steering vectors and magnitudes
    """
    X_base = get_activation_matrix(base_data, position, layer)
    X_instruct = get_activation_matrix(instruct_data, position, layer)

    base_cats = base_data.df['category'].values
    instruct_cats = instruct_data.df['category'].values

    category_steering = {}

    for cat in CATEGORIES:
        base_mask = base_cats == cat
        inst_mask = instruct_cats == cat

        if base_mask.sum() > 0 and inst_mask.sum() > 0:
            base_centroid = X_base[base_mask].mean(axis=0)
            inst_centroid = X_instruct[inst_mask].mean(axis=0)
            steering = inst_centroid - base_centroid

            category_steering[cat] = {
                'steering_vector': steering,
                'magnitude': float(np.linalg.norm(steering)),
                'base_centroid': base_centroid,
                'instruct_centroid': inst_centroid,
                'n_samples': int(base_mask.sum())
            }

    # Compute overall steering for reference
    overall_steering = X_instruct.mean(axis=0) - X_base.mean(axis=0)
    overall_mag = np.linalg.norm(overall_steering)

    results = {
        'category_steering': category_steering,
        'overall_steering': overall_steering,
        'overall_magnitude': float(overall_mag),
        'categories_found': list(category_steering.keys())
    }

    if print_output:
        print("\n" + "=" * 60)
        print("PER-SUBCATEGORY STEERING VECTORS")
        print("=" * 60)

        print(f"\n{'Category':<25} {'Magnitude':>12} {'N samples':>10}")
        print("-" * 50)

        for cat in CATEGORIES:
            if cat in category_steering:
                cs = category_steering[cat]
                marker = " [policy]" if cat in POLICY_CATEGORIES else ""
                print(f"{cat:<25} {cs['magnitude']:>12.3f} {cs['n_samples']:>10}{marker}")

        print("-" * 50)
        print(f"{'Overall (mean)':<25} {overall_mag:>12.3f}")

    return results


def subcategory_convergence_analysis(
    base_data: ModelData,
    instruct_data: ModelData,
    position: str = 'last',
    layer: Optional[int] = None,
    print_output: bool = True
) -> Dict:
    """
    Analyze convergence of subcategories during fine-tuning.

    Key test: If fine-tuning causes entanglement, policy categories
    (confident_incorrect, ambiguous, nonsensical) should have HIGH pairwise
    cosine similarity in their steering vectors - they're all moving toward
    similar representation spaces.

    We compare:
    1. Within-policy similarity: How similar are policy subcategory steering vectors?
    2. Within-factual similarity: How similar are factual subcategory steering vectors?
    3. Cross-group similarity: Policy vs factual steering similarity

    If policy categories are converging more than factual categories, that's
    direct evidence of trained-behavior entanglement.

    Args:
        base_data: ModelData for base model
        instruct_data: ModelData for instruct model
        position: Token position
        layer: Specific layer or None
        print_output: Whether to print results

    Returns:
        Dictionary with convergence analysis
    """
    # Get per-category steering vectors
    sv_results = extract_subcategory_steering_vectors(
        base_data, instruct_data, position, layer, print_output=False
    )
    category_steering = sv_results['category_steering']

    def cosine_sim(v1, v2):
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    # Compute pairwise similarities
    pairwise_sims = {}
    categories_present = list(category_steering.keys())

    for i, cat1 in enumerate(categories_present):
        for cat2 in categories_present[i+1:]:
            v1 = category_steering[cat1]['steering_vector']
            v2 = category_steering[cat2]['steering_vector']
            sim = cosine_sim(v1, v2)
            pairwise_sims[(cat1, cat2)] = sim

    # Compute group-level similarities
    policy_cats_present = [c for c in POLICY_CATEGORIES if c in category_steering]
    factual_cats_present = [c for c in FACTUAL_CATEGORIES if c in category_steering]

    # Within-policy similarities
    within_policy_sims = []
    for i, cat1 in enumerate(policy_cats_present):
        for cat2 in policy_cats_present[i+1:]:
            key = (cat1, cat2) if (cat1, cat2) in pairwise_sims else (cat2, cat1)
            if key in pairwise_sims:
                within_policy_sims.append(pairwise_sims[key])

    # Within-factual similarities
    within_factual_sims = []
    for i, cat1 in enumerate(factual_cats_present):
        for cat2 in factual_cats_present[i+1:]:
            key = (cat1, cat2) if (cat1, cat2) in pairwise_sims else (cat2, cat1)
            if key in pairwise_sims:
                within_factual_sims.append(pairwise_sims[key])

    # Cross-group similarities (policy to factual)
    cross_group_sims = []
    for cat1 in policy_cats_present:
        for cat2 in factual_cats_present:
            key = (cat1, cat2) if (cat1, cat2) in pairwise_sims else (cat2, cat1)
            if key in pairwise_sims:
                cross_group_sims.append(pairwise_sims[key])

    # Compute means
    within_policy_mean = float(np.mean(within_policy_sims)) if within_policy_sims else None
    within_factual_mean = float(np.mean(within_factual_sims)) if within_factual_sims else None
    cross_group_mean = float(np.mean(cross_group_sims)) if cross_group_sims else None

    # Convergence metric: how much MORE similar are policy subcategories
    # compared to cross-group baseline
    if within_policy_mean is not None and cross_group_mean is not None:
        policy_convergence = within_policy_mean - cross_group_mean
    else:
        policy_convergence = None

    if within_factual_mean is not None and cross_group_mean is not None:
        factual_convergence = within_factual_mean - cross_group_mean
    else:
        factual_convergence = None

    results = {
        'category_steering': {cat: {'magnitude': cs['magnitude'], 'n_samples': cs['n_samples']}
                              for cat, cs in category_steering.items()},
        'pairwise_similarities': {f"{k[0]} ↔ {k[1]}": v for k, v in pairwise_sims.items()},
        'within_policy_similarities': within_policy_sims,
        'within_factual_similarities': within_factual_sims,
        'cross_group_similarities': cross_group_sims,
        'summary': {
            'within_policy_mean_sim': within_policy_mean,
            'within_factual_mean_sim': within_factual_mean,
            'cross_group_mean_sim': cross_group_mean,
            'policy_convergence_excess': policy_convergence,
            'factual_convergence_excess': factual_convergence
        },
        'interpretation': ''
    }

    # Generate interpretation
    interp_parts = []

    if within_policy_mean is not None and within_policy_mean > 0.8:
        interp_parts.append(f"Policy subcategories are highly similar (mean sim={within_policy_mean:.2f})")
    elif within_policy_mean is not None and within_policy_mean > 0.6:
        interp_parts.append(f"Policy subcategories are moderately similar (mean sim={within_policy_mean:.2f})")

    if policy_convergence is not None and policy_convergence > 0.1:
        interp_parts.append(f"Policy categories converge +{policy_convergence:.2f} above cross-group baseline")

    if within_policy_mean is not None and within_factual_mean is not None:
        if within_policy_mean > within_factual_mean + 0.05:
            interp_parts.append(f"Policy converges MORE than factual ({within_policy_mean:.2f} vs {within_factual_mean:.2f})")
        elif within_factual_mean > within_policy_mean + 0.05:
            interp_parts.append(f"Factual converges MORE than policy ({within_factual_mean:.2f} vs {within_policy_mean:.2f})")

    if interp_parts:
        results['interpretation'] = " | ".join(interp_parts)
    else:
        results['interpretation'] = "No strong convergence pattern detected."

    if print_output:
        print("\n" + "=" * 60)
        print("SUBCATEGORY CONVERGENCE ANALYSIS")
        print("=" * 60)

        print("\nPairwise Steering Vector Similarities:")
        print(f"{'Category Pair':<45} {'Similarity':>12}")
        print("-" * 60)

        # Sort by similarity for readability
        sorted_pairs = sorted(pairwise_sims.items(), key=lambda x: x[1], reverse=True)
        for (cat1, cat2), sim in sorted_pairs:
            # Mark if both are policy or both are factual
            if cat1 in POLICY_CATEGORIES and cat2 in POLICY_CATEGORIES:
                marker = " [within-policy]"
            elif cat1 in FACTUAL_CATEGORIES and cat2 in FACTUAL_CATEGORIES:
                marker = " [within-factual]"
            else:
                marker = " [cross-group]"
            print(f"{cat1} ↔ {cat2:<20} {sim:>12.3f}{marker}")

        print("\n" + "-" * 60)
        print("Group Convergence Summary:")
        print(f"  Within-policy mean similarity:   {within_policy_mean:>10.3f}" if within_policy_mean else "  Within-policy: N/A")
        print(f"  Within-factual mean similarity:  {within_factual_mean:>10.3f}" if within_factual_mean else "  Within-factual: N/A")
        print(f"  Cross-group mean similarity:     {cross_group_mean:>10.3f}" if cross_group_mean else "  Cross-group: N/A")

        if policy_convergence is not None:
            print(f"\n  Policy convergence excess:       {policy_convergence:>+10.3f}")
        if factual_convergence is not None:
            print(f"  Factual convergence excess:      {factual_convergence:>+10.3f}")

        print(f"\nINTERPRETATION: {results['interpretation']}")

    return results


def subcategory_centroid_analysis(
    base_data: ModelData,
    instruct_data: ModelData,
    position: str = 'last',
    layer: Optional[int] = None,
    print_output: bool = True
) -> Dict:
    """
    Analyze how category centroids change during fine-tuning.

    Measures:
    1. How far each category centroid moves (displacement magnitude)
    2. Whether policy category centroids move toward each other (convergence)
    3. Change in within-group centroid distances

    Args:
        base_data: ModelData for base model
        instruct_data: ModelData for instruct model
        position: Token position
        layer: Specific layer or None
        print_output: Whether to print results

    Returns:
        Dictionary with centroid analysis
    """
    X_base = get_activation_matrix(base_data, position, layer)
    X_instruct = get_activation_matrix(instruct_data, position, layer)

    base_cats = base_data.df['category'].values
    instruct_cats = instruct_data.df['category'].values

    # Compute centroids for each category
    base_centroids = {}
    instruct_centroids = {}

    for cat in CATEGORIES:
        base_mask = base_cats == cat
        inst_mask = instruct_cats == cat

        if base_mask.sum() > 0 and inst_mask.sum() > 0:
            base_centroids[cat] = X_base[base_mask].mean(axis=0)
            instruct_centroids[cat] = X_instruct[inst_mask].mean(axis=0)

    categories_present = list(base_centroids.keys())

    # Compute displacement magnitudes
    displacements = {}
    for cat in categories_present:
        disp = instruct_centroids[cat] - base_centroids[cat]
        displacements[cat] = float(np.linalg.norm(disp))

    # Compute pairwise centroid distances (base and instruct)
    def euclidean_dist(v1, v2):
        return float(np.linalg.norm(v1 - v2))

    base_distances = {}
    instruct_distances = {}
    distance_changes = {}

    for i, cat1 in enumerate(categories_present):
        for cat2 in categories_present[i+1:]:
            base_dist = euclidean_dist(base_centroids[cat1], base_centroids[cat2])
            inst_dist = euclidean_dist(instruct_centroids[cat1], instruct_centroids[cat2])

            pair_key = f"{cat1} ↔ {cat2}"
            base_distances[pair_key] = base_dist
            instruct_distances[pair_key] = inst_dist
            distance_changes[pair_key] = {
                'base': base_dist,
                'instruct': inst_dist,
                'change': inst_dist - base_dist,
                'change_pct': (inst_dist - base_dist) / base_dist * 100 if base_dist > 0 else 0
            }

    # Aggregate by group
    policy_cats_present = [c for c in POLICY_CATEGORIES if c in categories_present]
    factual_cats_present = [c for c in FACTUAL_CATEGORIES if c in categories_present]

    # Within-policy distance changes
    within_policy_changes = []
    for i, cat1 in enumerate(policy_cats_present):
        for cat2 in policy_cats_present[i+1:]:
            pair_key = f"{cat1} ↔ {cat2}"
            if pair_key in distance_changes:
                within_policy_changes.append(distance_changes[pair_key]['change_pct'])

    # Within-factual distance changes
    within_factual_changes = []
    for i, cat1 in enumerate(factual_cats_present):
        for cat2 in factual_cats_present[i+1:]:
            pair_key = f"{cat1} ↔ {cat2}"
            if pair_key in distance_changes:
                within_factual_changes.append(distance_changes[pair_key]['change_pct'])

    # Cross-group distance changes
    cross_group_changes = []
    for cat1 in policy_cats_present:
        for cat2 in factual_cats_present:
            pair_key = f"{cat1} ↔ {cat2}"
            if pair_key in distance_changes:
                cross_group_changes.append(distance_changes[pair_key]['change_pct'])
            else:
                # Try reverse order
                pair_key = f"{cat2} ↔ {cat1}"
                if pair_key in distance_changes:
                    cross_group_changes.append(distance_changes[pair_key]['change_pct'])

    within_policy_mean_change = float(np.mean(within_policy_changes)) if within_policy_changes else None
    within_factual_mean_change = float(np.mean(within_factual_changes)) if within_factual_changes else None
    cross_group_mean_change = float(np.mean(cross_group_changes)) if cross_group_changes else None

    results = {
        'displacements': displacements,
        'distance_changes': distance_changes,
        'summary': {
            'within_policy_mean_change_pct': within_policy_mean_change,
            'within_factual_mean_change_pct': within_factual_mean_change,
            'cross_group_mean_change_pct': cross_group_mean_change,
            'policy_displacement_mean': float(np.mean([displacements[c] for c in policy_cats_present])) if policy_cats_present else None,
            'factual_displacement_mean': float(np.mean([displacements[c] for c in factual_cats_present])) if factual_cats_present else None
        },
        'interpretation': ''
    }

    # Generate interpretation
    interp_parts = []

    if within_policy_mean_change is not None and within_policy_mean_change < -5:
        interp_parts.append(f"Policy centroids CONVERGE ({within_policy_mean_change:+.1f}% distance)")
    elif within_policy_mean_change is not None and within_policy_mean_change > 5:
        interp_parts.append(f"Policy centroids DIVERGE ({within_policy_mean_change:+.1f}% distance)")

    if within_factual_mean_change is not None and within_factual_mean_change < -5:
        interp_parts.append(f"Factual centroids CONVERGE ({within_factual_mean_change:+.1f}% distance)")
    elif within_factual_mean_change is not None and within_factual_mean_change > 5:
        interp_parts.append(f"Factual centroids DIVERGE ({within_factual_mean_change:+.1f}% distance)")

    if cross_group_mean_change is not None and abs(cross_group_mean_change) > 5:
        direction = "INCREASE" if cross_group_mean_change > 0 else "DECREASE"
        interp_parts.append(f"Cross-group distance {direction}s ({cross_group_mean_change:+.1f}%)")

    # Key comparison: do policy categories converge more than factual?
    if within_policy_mean_change is not None and within_factual_mean_change is not None:
        if within_policy_mean_change < within_factual_mean_change - 5:
            interp_parts.append("Policy converges MORE than factual")
        elif within_factual_mean_change < within_policy_mean_change - 5:
            interp_parts.append("Factual converges MORE than policy")

    results['interpretation'] = " | ".join(interp_parts) if interp_parts else "No strong convergence/divergence pattern."

    if print_output:
        print("\n" + "=" * 60)
        print("SUBCATEGORY CENTROID ANALYSIS")
        print("=" * 60)

        print("\nCentroid Displacement Magnitudes:")
        print(f"{'Category':<25} {'Displacement':>12}")
        print("-" * 40)
        for cat in CATEGORIES:
            if cat in displacements:
                marker = " [policy]" if cat in POLICY_CATEGORIES else ""
                print(f"{cat:<25} {displacements[cat]:>12.3f}{marker}")

        print("\n" + "-" * 60)
        print("Pairwise Centroid Distance Changes:")
        print(f"{'Category Pair':<40} {'Base':>10} {'Instruct':>10} {'Change':>10}")
        print("-" * 75)

        # Sort by change percentage
        sorted_changes = sorted(distance_changes.items(), key=lambda x: x[1]['change_pct'])
        for pair_key, dc in sorted_changes:
            cats = pair_key.split(" ↔ ")
            # Determine group
            if cats[0] in POLICY_CATEGORIES and cats[1] in POLICY_CATEGORIES:
                marker = " [within-policy]"
            elif cats[0] in FACTUAL_CATEGORIES and cats[1] in FACTUAL_CATEGORIES:
                marker = " [within-factual]"
            else:
                marker = ""
            print(f"{pair_key:<40} {dc['base']:>10.1f} {dc['instruct']:>10.1f} {dc['change_pct']:>+9.1f}%{marker}")

        print("\n" + "-" * 60)
        print("Group Summary (mean % change in pairwise distances):")
        if within_policy_mean_change is not None:
            sign = "↓ CONVERGE" if within_policy_mean_change < 0 else "↑ DIVERGE"
            print(f"  Within-policy:   {within_policy_mean_change:>+8.1f}%  {sign}")
        if within_factual_mean_change is not None:
            sign = "↓ CONVERGE" if within_factual_mean_change < 0 else "↑ DIVERGE"
            print(f"  Within-factual:  {within_factual_mean_change:>+8.1f}%  {sign}")
        if cross_group_mean_change is not None:
            sign = "↑ SEPARATE" if cross_group_mean_change > 0 else "↓ MERGE"
            print(f"  Cross-group:     {cross_group_mean_change:>+8.1f}%  {sign}")

        print(f"\nINTERPRETATION: {results['interpretation']}")

    return results


def run_subcategory_convergence_analysis(
    base_data: ModelData,
    instruct_data: ModelData,
    position: str = 'last',
    print_output: bool = True
) -> Dict:
    """
    Run complete subcategory convergence analysis suite.

    Tests whether fine-tuning causes policy subcategories to converge
    toward similar representation spaces (the entanglement hypothesis).

    Combines:
    1. Per-subcategory steering vectors
    2. Steering vector similarity analysis (do they point same direction?)
    3. Centroid distance analysis (do they move closer together?)

    Args:
        base_data: ModelData for base model
        instruct_data: ModelData for instruct model
        position: Token position
        print_output: Whether to print results

    Returns:
        Combined results dictionary
    """
    if print_output:
        print("\n" + "=" * 70)
        print(f"SUBCATEGORY CONVERGENCE: {base_data.name} -> {instruct_data.name}")
        print("=" * 70)

    # 1. Per-category steering vectors
    sv_results = extract_subcategory_steering_vectors(
        base_data, instruct_data, position, print_output=print_output
    )

    # 2. Convergence analysis (steering vector similarities)
    convergence_results = subcategory_convergence_analysis(
        base_data, instruct_data, position, print_output=print_output
    )

    # 3. Centroid distance analysis
    centroid_results = subcategory_centroid_analysis(
        base_data, instruct_data, position, print_output=print_output
    )

    results = {
        'base_model': base_data.name,
        'instruct_model': instruct_data.name,
        'position': position,
        'steering_vectors': {
            'magnitudes': {cat: sv_results['category_steering'][cat]['magnitude']
                          for cat in sv_results['category_steering']}
        },
        'convergence': {
            'pairwise_similarities': convergence_results['pairwise_similarities'],
            'summary': convergence_results['summary']
        },
        'centroids': {
            'displacements': centroid_results['displacements'],
            'summary': centroid_results['summary']
        },
        'overall_interpretation': ''
    }

    # Generate overall interpretation
    findings = []

    # Steering vector convergence
    conv_sum = convergence_results['summary']
    if conv_sum['within_policy_mean_sim'] is not None:
        if conv_sum['within_policy_mean_sim'] > 0.8:
            findings.append(f"Policy steering vectors highly aligned (sim={conv_sum['within_policy_mean_sim']:.2f})")

        if conv_sum['policy_convergence_excess'] is not None and conv_sum['policy_convergence_excess'] > 0.1:
            findings.append(f"Policy converges +{conv_sum['policy_convergence_excess']:.2f} above cross-group")

    # Centroid distance changes
    cent_sum = centroid_results['summary']
    if cent_sum['within_policy_mean_change_pct'] is not None:
        if cent_sum['within_policy_mean_change_pct'] < -10:
            findings.append(f"Policy centroids move {cent_sum['within_policy_mean_change_pct']:.0f}% closer")

    # Compare policy vs factual convergence
    if (conv_sum['within_policy_mean_sim'] is not None and
        conv_sum['within_factual_mean_sim'] is not None):
        if conv_sum['within_policy_mean_sim'] > conv_sum['within_factual_mean_sim'] + 0.1:
            findings.append("Policy converges MORE than factual in steering direction")

    if (cent_sum['within_policy_mean_change_pct'] is not None and
        cent_sum['within_factual_mean_change_pct'] is not None):
        if cent_sum['within_policy_mean_change_pct'] < cent_sum['within_factual_mean_change_pct'] - 5:
            findings.append("Policy centroids converge MORE than factual")

    if findings:
        results['overall_interpretation'] = (
            f"EVIDENCE FOR POLICY CONVERGENCE: {'; '.join(findings)}. "
            f"Fine-tuning causes policy categories (hallucination acknowledgment, "
            f"ambiguity recognition, nonsensical detection) to move toward similar "
            f"representation spaces - this is the entanglement mechanism."
        )
    else:
        results['overall_interpretation'] = (
            "No strong evidence for differential convergence. Policy and factual "
            "categories may change similarly during fine-tuning."
        )

    if print_output:
        print("\n" + "=" * 70)
        print("OVERALL CONVERGENCE SUMMARY")
        print("=" * 70)
        print(f"\n{results['overall_interpretation']}")

    _cleanup()
    return results
