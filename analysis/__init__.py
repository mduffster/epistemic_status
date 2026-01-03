"""
Epistemic Probing Analysis Package

Modular analysis tools for understanding epistemic states in language models.
"""

from .loader import load_model_data, ModelData
from .core import basic_stats, failure_mode_analysis, analyze_prompt_features
from .probing import run_linear_probe, run_mlp_probe, layer_analysis, compare_positions, probe_with_controls, compare_linear_vs_mlp
from .entropy import entropy_analysis, entropy_vs_probe
from .calibration import confidence_calibration
from .effects import compute_effect_sizes, compute_roc_auc
from .comparison import cross_model_generalization, compare_models, transfer_by_category
from .statistics import (
    # Significance testing
    permutation_test,
    bootstrap_paired_test,
    probe_permutation_test,
    SignificanceResult,
    # Multiple comparison correction
    bonferroni_correction,
    holm_correction,
    fdr_correction,
    correct_multiple_comparisons,
    MultipleComparisonResult,
    # Seed sensitivity
    run_seed_sensitivity,
    run_probe_seed_sensitivity,
    SeedSensitivityResult,
    # Utilities
    summarize_significance,
)

__all__ = [
    # Data loading
    'load_model_data',
    'ModelData',
    # Core analysis
    'basic_stats',
    'failure_mode_analysis',
    'analyze_prompt_features',
    # Probing
    'run_linear_probe',
    'run_mlp_probe',
    'layer_analysis',
    'compare_positions',
    'probe_with_controls',
    'compare_linear_vs_mlp',
    # Entropy
    'entropy_analysis',
    'entropy_vs_probe',
    # Calibration
    'confidence_calibration',
    # Effects
    'compute_effect_sizes',
    'compute_roc_auc',
    # Cross-model comparison
    'cross_model_generalization',
    'compare_models',
    'transfer_by_category',
    # Statistical significance (NEW)
    'permutation_test',
    'bootstrap_paired_test',
    'probe_permutation_test',
    'SignificanceResult',
    # Multiple comparison correction (NEW)
    'bonferroni_correction',
    'holm_correction',
    'fdr_correction',
    'correct_multiple_comparisons',
    'MultipleComparisonResult',
    # Seed sensitivity (NEW)
    'run_seed_sensitivity',
    'run_probe_seed_sensitivity',
    'SeedSensitivityResult',
    # Utilities (NEW)
    'summarize_significance',
]
