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
from .steering import (
    extract_steering_vector,
    project_onto_steering,
    steering_by_category,
    low_rank_analysis,
    ablate_steering_subspace,
    run_full_steering_analysis,
    # Category-specific steering
    extract_category_steering_vectors,
    category_specific_projection_analysis,
    ablate_differential_steering,
    run_category_specific_steering_analysis,
    # Subcategory convergence analysis
    extract_subcategory_steering_vectors,
    subcategory_convergence_analysis,
    subcategory_centroid_analysis,
    run_subcategory_convergence_analysis,
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
    # Steering vector analysis (NEW)
    'extract_steering_vector',
    'project_onto_steering',
    'steering_by_category',
    'low_rank_analysis',
    'ablate_steering_subspace',
    'run_full_steering_analysis',
    # Category-specific steering (NEW)
    'extract_category_steering_vectors',
    'category_specific_projection_analysis',
    'ablate_differential_steering',
    'run_category_specific_steering_analysis',
    # Subcategory convergence analysis (NEW)
    'extract_subcategory_steering_vectors',
    'subcategory_convergence_analysis',
    'subcategory_centroid_analysis',
    'run_subcategory_convergence_analysis',
]
