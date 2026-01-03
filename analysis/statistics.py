"""
Statistical significance testing and multiple comparison correction.

Provides rigorous hypothesis testing for epistemic probing analysis:
- Permutation tests for comparing groups/models
- Bootstrap significance tests with proper p-values
- Multiple comparison correction (Bonferroni, Holm, FDR)
- Seed sensitivity analysis utilities
"""

from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings


@dataclass
class SignificanceResult:
    """Container for significance test results."""
    statistic: float
    p_value: float
    ci_low: float
    ci_high: float
    effect_size: float
    n_permutations: int
    significant: bool
    alpha: float

    def __repr__(self):
        sig_str = "***" if self.p_value < 0.001 else "**" if self.p_value < 0.01 else "*" if self.p_value < 0.05 else ""
        return (f"SignificanceResult(stat={self.statistic:.4f}, p={self.p_value:.4f}{sig_str}, "
                f"95% CI=[{self.ci_low:.4f}, {self.ci_high:.4f}], d={self.effect_size:.3f})")


@dataclass
class MultipleComparisonResult:
    """Container for multiple comparison corrected results."""
    original_p_values: np.ndarray
    corrected_p_values: np.ndarray
    significant: np.ndarray
    method: str
    alpha: float
    n_tests: int
    n_significant: int

    def to_dataframe(self, labels: Optional[List[str]] = None) -> pd.DataFrame:
        """Convert to DataFrame for easy viewing."""
        df = pd.DataFrame({
            'p_original': self.original_p_values,
            'p_corrected': self.corrected_p_values,
            'significant': self.significant
        })
        if labels:
            df.index = labels
        return df


def permutation_test(
    group1: np.ndarray,
    group2: np.ndarray,
    statistic: str = 'mean_diff',
    n_permutations: int = 10000,
    alternative: str = 'two-sided',
    random_state: int = 42
) -> SignificanceResult:
    """
    Permutation test for comparing two groups.

    Args:
        group1: First group of observations
        group2: Second group of observations
        statistic: 'mean_diff', 'median_diff', or 'cohens_d'
        n_permutations: Number of permutations
        alternative: 'two-sided', 'greater', or 'less'
        random_state: Random seed

    Returns:
        SignificanceResult with p-value and effect size
    """
    rng = np.random.RandomState(random_state)

    group1 = np.asarray(group1).flatten()
    group2 = np.asarray(group2).flatten()

    n1, n2 = len(group1), len(group2)
    combined = np.concatenate([group1, group2])

    # Compute observed statistic
    if statistic == 'mean_diff':
        observed = group1.mean() - group2.mean()
        stat_func = lambda g1, g2: g1.mean() - g2.mean()
    elif statistic == 'median_diff':
        observed = np.median(group1) - np.median(group2)
        stat_func = lambda g1, g2: np.median(g1) - np.median(g2)
    elif statistic == 'cohens_d':
        observed = _cohens_d(group1, group2)
        stat_func = _cohens_d
    else:
        raise ValueError(f"Unknown statistic: {statistic}")

    # Permutation distribution
    perm_stats = np.zeros(n_permutations)
    for i in range(n_permutations):
        perm = rng.permutation(combined)
        perm_g1 = perm[:n1]
        perm_g2 = perm[n1:]
        perm_stats[i] = stat_func(perm_g1, perm_g2)

    # Compute p-value
    if alternative == 'two-sided':
        p_value = np.mean(np.abs(perm_stats) >= np.abs(observed))
    elif alternative == 'greater':
        p_value = np.mean(perm_stats >= observed)
    elif alternative == 'less':
        p_value = np.mean(perm_stats <= observed)
    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    # Bootstrap CI for the observed statistic
    ci_low, ci_high = _bootstrap_ci(group1, group2, stat_func, random_state=random_state)

    # Effect size (always Cohen's d for standardized comparison)
    effect_size = _cohens_d(group1, group2)

    return SignificanceResult(
        statistic=observed,
        p_value=p_value,
        ci_low=ci_low,
        ci_high=ci_high,
        effect_size=effect_size,
        n_permutations=n_permutations,
        significant=p_value < 0.05,
        alpha=0.05
    )


def bootstrap_paired_test(
    values1: np.ndarray,
    values2: np.ndarray,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    random_state: int = 42
) -> SignificanceResult:
    """
    Bootstrap test for paired samples (e.g., same samples, two models).

    Tests H0: mean(values1 - values2) = 0

    Args:
        values1: First set of paired values
        values2: Second set of paired values (same length)
        n_bootstrap: Number of bootstrap iterations
        confidence_level: Confidence level for CI
        random_state: Random seed

    Returns:
        SignificanceResult with p-value for difference from zero
    """
    rng = np.random.RandomState(random_state)

    values1 = np.asarray(values1).flatten()
    values2 = np.asarray(values2).flatten()

    if len(values1) != len(values2):
        raise ValueError("Paired test requires equal length arrays")

    n = len(values1)
    differences = values1 - values2
    observed_diff = differences.mean()

    # Bootstrap under null hypothesis (centered differences)
    centered_diff = differences - observed_diff

    boot_means = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        boot_idx = rng.randint(0, n, size=n)
        boot_means[i] = centered_diff[boot_idx].mean()

    # Two-sided p-value
    p_value = np.mean(np.abs(boot_means) >= np.abs(observed_diff))

    # Bootstrap CI for observed difference
    boot_diffs = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        boot_idx = rng.randint(0, n, size=n)
        boot_diffs[i] = differences[boot_idx].mean()

    alpha = 1 - confidence_level
    ci_low = np.percentile(boot_diffs, alpha/2 * 100)
    ci_high = np.percentile(boot_diffs, (1 - alpha/2) * 100)

    # Effect size: standardized mean difference
    effect_size = observed_diff / (differences.std() + 1e-10)

    return SignificanceResult(
        statistic=observed_diff,
        p_value=p_value,
        ci_low=ci_low,
        ci_high=ci_high,
        effect_size=effect_size,
        n_permutations=n_bootstrap,
        significant=p_value < 0.05,
        alpha=0.05
    )


def probe_permutation_test(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    group1_value: Any,
    group2_value: Any,
    metric: str = 'accuracy',
    n_permutations: int = 1000,
    test_size: float = 0.2,
    random_state: int = 42
) -> SignificanceResult:
    """
    Permutation test for probe performance difference between groups.

    Tests whether probe accuracy/AUC differs between two category groups.

    Args:
        X: Feature matrix
        y: Labels (correct/incorrect)
        groups: Group assignments for each sample
        group1_value: Value identifying first group
        group2_value: Value identifying second group
        metric: 'accuracy' or 'auc'
        n_permutations: Number of permutations
        test_size: Fraction for test set
        random_state: Random seed

    Returns:
        SignificanceResult comparing probe performance between groups
    """
    rng = np.random.RandomState(random_state)

    # Get observed difference
    def compute_group_metric(X, y, groups, g1_val, g2_val, rs):
        X_train, X_test, y_train, y_test, grp_train, grp_test = train_test_split(
            X, y, groups, test_size=test_size, random_state=rs, stratify=y
        )

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=1000, random_state=rs)
        clf.fit(X_train_s, y_train)

        # Get metrics for each group on test set
        g1_mask = grp_test == g1_val
        g2_mask = grp_test == g2_val

        if g1_mask.sum() < 2 or g2_mask.sum() < 2:
            return np.nan

        if metric == 'accuracy':
            preds = clf.predict(X_test_s)
            g1_metric = (preds[g1_mask] == y_test[g1_mask]).mean()
            g2_metric = (preds[g2_mask] == y_test[g2_mask]).mean()
        else:  # AUC
            probs = clf.predict_proba(X_test_s)[:, 1]
            try:
                g1_metric = roc_auc_score(y_test[g1_mask], probs[g1_mask])
                g2_metric = roc_auc_score(y_test[g2_mask], probs[g2_mask])
            except ValueError:
                return np.nan

        return g1_metric - g2_metric

    observed = compute_group_metric(X, y, groups, group1_value, group2_value, random_state)

    if np.isnan(observed):
        return SignificanceResult(
            statistic=np.nan, p_value=1.0, ci_low=np.nan, ci_high=np.nan,
            effect_size=np.nan, n_permutations=0, significant=False, alpha=0.05
        )

    # Permutation test: shuffle group labels
    perm_diffs = []
    for i in range(n_permutations):
        perm_groups = rng.permutation(groups)
        perm_diff = compute_group_metric(X, y, perm_groups, group1_value, group2_value, random_state + i)
        if not np.isnan(perm_diff):
            perm_diffs.append(perm_diff)

    perm_diffs = np.array(perm_diffs)

    if len(perm_diffs) < 100:
        warnings.warn(f"Only {len(perm_diffs)} valid permutations")

    p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed))

    # Bootstrap CI
    boot_diffs = []
    for i in range(min(n_permutations, 500)):
        boot_diff = compute_group_metric(X, y, groups, group1_value, group2_value, random_state + i + n_permutations)
        if not np.isnan(boot_diff):
            boot_diffs.append(boot_diff)

    boot_diffs = np.array(boot_diffs)
    ci_low = np.percentile(boot_diffs, 2.5) if len(boot_diffs) > 0 else np.nan
    ci_high = np.percentile(boot_diffs, 97.5) if len(boot_diffs) > 0 else np.nan

    return SignificanceResult(
        statistic=observed,
        p_value=p_value,
        ci_low=ci_low,
        ci_high=ci_high,
        effect_size=observed,  # The difference itself is the effect
        n_permutations=len(perm_diffs),
        significant=p_value < 0.05,
        alpha=0.05
    )


# =============================================================================
# Multiple Comparison Correction
# =============================================================================

def bonferroni_correction(
    p_values: np.ndarray,
    alpha: float = 0.05
) -> MultipleComparisonResult:
    """
    Bonferroni correction for multiple comparisons.

    Most conservative method - controls family-wise error rate (FWER).

    Args:
        p_values: Array of p-values
        alpha: Significance level

    Returns:
        MultipleComparisonResult with corrected p-values
    """
    p_values = np.asarray(p_values)
    n_tests = len(p_values)

    corrected = np.minimum(p_values * n_tests, 1.0)
    significant = corrected < alpha

    return MultipleComparisonResult(
        original_p_values=p_values,
        corrected_p_values=corrected,
        significant=significant,
        method='bonferroni',
        alpha=alpha,
        n_tests=n_tests,
        n_significant=significant.sum()
    )


def holm_correction(
    p_values: np.ndarray,
    alpha: float = 0.05
) -> MultipleComparisonResult:
    """
    Holm-Bonferroni step-down correction.

    Less conservative than Bonferroni while still controlling FWER.

    Args:
        p_values: Array of p-values
        alpha: Significance level

    Returns:
        MultipleComparisonResult with corrected p-values
    """
    p_values = np.asarray(p_values)
    n_tests = len(p_values)

    # Sort p-values
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]

    # Holm correction: p_i * (n - i + 1)
    corrected_sorted = np.zeros(n_tests)
    for i in range(n_tests):
        corrected_sorted[i] = sorted_p[i] * (n_tests - i)

    # Enforce monotonicity (corrected p-values should be non-decreasing)
    for i in range(1, n_tests):
        corrected_sorted[i] = max(corrected_sorted[i], corrected_sorted[i-1])

    corrected_sorted = np.minimum(corrected_sorted, 1.0)

    # Restore original order
    corrected = np.zeros(n_tests)
    corrected[sorted_idx] = corrected_sorted

    significant = corrected < alpha

    return MultipleComparisonResult(
        original_p_values=p_values,
        corrected_p_values=corrected,
        significant=significant,
        method='holm',
        alpha=alpha,
        n_tests=n_tests,
        n_significant=significant.sum()
    )


def fdr_correction(
    p_values: np.ndarray,
    alpha: float = 0.05,
    method: str = 'bh'
) -> MultipleComparisonResult:
    """
    False Discovery Rate (FDR) correction.

    Controls the expected proportion of false positives among rejections.
    Less conservative than FWER methods.

    Args:
        p_values: Array of p-values
        alpha: Significance level (target FDR)
        method: 'bh' (Benjamini-Hochberg) or 'by' (Benjamini-Yekutieli)

    Returns:
        MultipleComparisonResult with corrected p-values
    """
    p_values = np.asarray(p_values)
    n_tests = len(p_values)

    # Sort p-values
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]

    # BH or BY adjustment factor
    if method == 'bh':
        # Benjamini-Hochberg
        factor = 1.0
    elif method == 'by':
        # Benjamini-Yekutieli (more conservative, valid under arbitrary dependence)
        factor = np.sum(1.0 / np.arange(1, n_tests + 1))
    else:
        raise ValueError(f"Unknown FDR method: {method}")

    # Compute adjusted p-values
    corrected_sorted = np.zeros(n_tests)
    for i in range(n_tests):
        corrected_sorted[i] = sorted_p[i] * n_tests * factor / (i + 1)

    # Enforce monotonicity (going backwards)
    for i in range(n_tests - 2, -1, -1):
        corrected_sorted[i] = min(corrected_sorted[i], corrected_sorted[i + 1])

    corrected_sorted = np.minimum(corrected_sorted, 1.0)

    # Restore original order
    corrected = np.zeros(n_tests)
    corrected[sorted_idx] = corrected_sorted

    significant = corrected < alpha

    return MultipleComparisonResult(
        original_p_values=p_values,
        corrected_p_values=corrected,
        significant=significant,
        method=f'fdr_{method}',
        alpha=alpha,
        n_tests=n_tests,
        n_significant=significant.sum()
    )


def correct_multiple_comparisons(
    p_values: np.ndarray,
    method: str = 'fdr_bh',
    alpha: float = 0.05,
    labels: Optional[List[str]] = None,
    print_output: bool = True
) -> MultipleComparisonResult:
    """
    Apply multiple comparison correction with optional output.

    Args:
        p_values: Array of p-values
        method: 'bonferroni', 'holm', 'fdr_bh', or 'fdr_by'
        alpha: Significance level
        labels: Optional labels for each test
        print_output: Whether to print results

    Returns:
        MultipleComparisonResult
    """
    if method == 'bonferroni':
        result = bonferroni_correction(p_values, alpha)
    elif method == 'holm':
        result = holm_correction(p_values, alpha)
    elif method == 'fdr_bh':
        result = fdr_correction(p_values, alpha, 'bh')
    elif method == 'fdr_by':
        result = fdr_correction(p_values, alpha, 'by')
    else:
        raise ValueError(f"Unknown method: {method}")

    if print_output:
        print(f"\n{'='*60}")
        print(f"MULTIPLE COMPARISON CORRECTION ({method.upper()})")
        print(f"{'='*60}")
        print(f"Number of tests: {result.n_tests}")
        print(f"Alpha level: {alpha}")
        print(f"Significant after correction: {result.n_significant}/{result.n_tests}")
        print()

        df = result.to_dataframe(labels)
        print(df.to_string())

    return result


# =============================================================================
# Seed Sensitivity Analysis
# =============================================================================

@dataclass
class SeedSensitivityResult:
    """Container for seed sensitivity analysis results."""
    seeds: List[int]
    metrics: Dict[str, np.ndarray]  # metric_name -> array of values per seed
    summary: Dict[str, Dict[str, float]]  # metric_name -> {mean, std, min, max, cv}
    is_stable: Dict[str, bool]  # metric_name -> whether CV < threshold
    stability_threshold: float

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame."""
        rows = []
        for metric, values in self.metrics.items():
            rows.append({
                'metric': metric,
                'mean': values.mean(),
                'std': values.std(),
                'min': values.min(),
                'max': values.max(),
                'cv': values.std() / (values.mean() + 1e-10),
                'stable': self.is_stable[metric]
            })
        return pd.DataFrame(rows)


def run_seed_sensitivity(
    analysis_func: Callable,
    seeds: List[int] = None,
    n_seeds: int = 5,
    base_seed: int = 42,
    extract_metrics: Callable = None,
    stability_threshold: float = 0.05,
    print_output: bool = True,
    **kwargs
) -> SeedSensitivityResult:
    """
    Run analysis with multiple random seeds to assess stability.

    Args:
        analysis_func: Function to run (should accept random_state kwarg)
        seeds: Specific seeds to use, or None to generate
        n_seeds: Number of seeds if not specified
        base_seed: Base for generating seeds
        extract_metrics: Function to extract metrics from analysis result
        stability_threshold: CV threshold for stability (default 5%)
        print_output: Whether to print results
        **kwargs: Additional kwargs passed to analysis_func

    Returns:
        SeedSensitivityResult with per-seed metrics and stability assessment
    """
    if seeds is None:
        rng = np.random.RandomState(base_seed)
        seeds = rng.randint(0, 10000, size=n_seeds).tolist()

    if extract_metrics is None:
        # Default: assume result is a dict with numeric values
        extract_metrics = lambda r: {k: v for k, v in r.items() if isinstance(v, (int, float))}

    all_metrics = {}

    if print_output:
        print(f"\n{'='*60}")
        print(f"SEED SENSITIVITY ANALYSIS")
        print(f"{'='*60}")
        print(f"Seeds: {seeds}")
        print()

    for seed in seeds:
        result = analysis_func(random_state=seed, print_output=False, **kwargs)
        metrics = extract_metrics(result)

        for metric, value in metrics.items():
            if metric not in all_metrics:
                all_metrics[metric] = []
            all_metrics[metric].append(value)

        if print_output:
            print(f"Seed {seed}: ", end="")
            print(", ".join(f"{k}={v:.4f}" for k, v in list(metrics.items())[:3]))

    # Convert to arrays and compute summary
    metrics_arrays = {k: np.array(v) for k, v in all_metrics.items()}

    summary = {}
    is_stable = {}

    for metric, values in metrics_arrays.items():
        mean = values.mean()
        std = values.std()
        cv = std / (mean + 1e-10)

        summary[metric] = {
            'mean': mean,
            'std': std,
            'min': values.min(),
            'max': values.max(),
            'cv': cv
        }
        is_stable[metric] = cv < stability_threshold

    result = SeedSensitivityResult(
        seeds=seeds,
        metrics=metrics_arrays,
        summary=summary,
        is_stable=is_stable,
        stability_threshold=stability_threshold
    )

    if print_output:
        print(f"\n--- Summary ---")
        df = result.to_dataframe()
        print(df.to_string(index=False))

        unstable = [m for m, stable in is_stable.items() if not stable]
        if unstable:
            print(f"\n⚠️  Unstable metrics (CV > {stability_threshold*100:.0f}%): {unstable}")
        else:
            print(f"\n✓ All metrics stable (CV < {stability_threshold*100:.0f}%)")

    return result


def run_probe_seed_sensitivity(
    X: np.ndarray,
    y: np.ndarray,
    n_seeds: int = 5,
    test_size: float = 0.2,
    print_output: bool = True
) -> SeedSensitivityResult:
    """
    Convenience function for probe seed sensitivity analysis.

    Args:
        X: Feature matrix
        y: Labels
        n_seeds: Number of seeds
        test_size: Test set fraction
        print_output: Whether to print results

    Returns:
        SeedSensitivityResult for probe accuracy and AUC
    """
    def run_probe(random_state, print_output=False):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=1000, random_state=random_state)
        clf.fit(X_train_s, y_train)

        acc = clf.score(X_test_s, y_test)
        probs = clf.predict_proba(X_test_s)[:, 1]

        try:
            auc = roc_auc_score(y_test, probs)
        except ValueError:
            auc = np.nan

        return {'accuracy': acc, 'auc': auc}

    return run_seed_sensitivity(
        run_probe,
        n_seeds=n_seeds,
        print_output=print_output
    )


# =============================================================================
# Helper Functions
# =============================================================================

def _cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()

    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std < 1e-10:
        return 0.0

    return (group1.mean() - group2.mean()) / pooled_std


def _bootstrap_ci(
    group1: np.ndarray,
    group2: np.ndarray,
    stat_func: Callable,
    n_bootstrap: int = 2000,
    confidence_level: float = 0.95,
    random_state: int = 42
) -> Tuple[float, float]:
    """Compute bootstrap CI for a two-sample statistic."""
    rng = np.random.RandomState(random_state)

    n1, n2 = len(group1), len(group2)
    boot_stats = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        boot_g1 = group1[rng.randint(0, n1, size=n1)]
        boot_g2 = group2[rng.randint(0, n2, size=n2)]
        boot_stats[i] = stat_func(boot_g1, boot_g2)

    alpha = 1 - confidence_level
    ci_low = np.percentile(boot_stats, alpha/2 * 100)
    ci_high = np.percentile(boot_stats, (1 - alpha/2) * 100)

    return ci_low, ci_high


def summarize_significance(
    results: Dict[str, SignificanceResult],
    correction_method: str = 'fdr_bh',
    alpha: float = 0.05,
    print_output: bool = True
) -> pd.DataFrame:
    """
    Summarize multiple significance tests with correction.

    Args:
        results: Dict mapping test names to SignificanceResult objects
        correction_method: Multiple comparison correction method
        alpha: Significance level
        print_output: Whether to print results

    Returns:
        DataFrame with summary
    """
    labels = list(results.keys())
    p_values = np.array([r.p_value for r in results.values()])

    # Apply correction
    corrected = correct_multiple_comparisons(
        p_values, method=correction_method, alpha=alpha,
        labels=labels, print_output=False
    )

    # Build summary DataFrame
    rows = []
    for i, (name, result) in enumerate(results.items()):
        rows.append({
            'test': name,
            'statistic': result.statistic,
            'p_value': result.p_value,
            'p_corrected': corrected.corrected_p_values[i],
            'significant': corrected.significant[i],
            'effect_size': result.effect_size,
            'ci_low': result.ci_low,
            'ci_high': result.ci_high
        })

    df = pd.DataFrame(rows)

    if print_output:
        print(f"\n{'='*80}")
        print(f"SIGNIFICANCE SUMMARY (correction: {correction_method})")
        print(f"{'='*80}")
        print(df.to_string(index=False))
        print(f"\nSignificant tests: {corrected.n_significant}/{corrected.n_tests}")

    return df
